// test_moe_pager.cpp -- standalone verification for the O_DIRECT expert pager.
//
// Proves the thing that matters before it is wired into the graph: the bytes
// the pager DMAs for a given (layer, expert) are byte-identical to the bytes
// sitting at that expert's slice in the GGUF.
//
// It does this by reading the same slice twice through two independent paths
// (a plain buffered pread, and the pager's O_DIRECT+io_uring path) and comparing.
#include "moe_pager.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cerrno>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

static const char * MODEL =
    "/mnt/windows/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS/IQ2_XS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf";

static bool read_reference(int fd, const moe_expert_slice & slice, void * destination, size_t bytes) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    const int buffered = open(path, O_RDONLY);
    if (buffered < 0) return false;
    size_t got = 0;
    while (got < bytes) {
        const ssize_t result = pread(buffered, static_cast<char *>(destination) + got, bytes - got, (off_t) (slice.off + slice.pre + got));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) break;
        got += (size_t) result;
    }
    close(buffered);
    return got == bytes;
}

static bool write_temp_index(const std::string & contents, std::string & path) {
    char pattern[] = "/tmp/betelgeuse-pack-index-XXXXXX";
    const int fd = mkstemp(pattern);
    if (fd < 0) return false;
    size_t written = 0;
    while (written < contents.size()) {
        const ssize_t n = write(fd, contents.data() + written, contents.size() - written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); unlink(pattern); return false; }
        written += (size_t) n;
    }
    if (close(fd) != 0) { unlink(pattern); return false; }
    path = pattern;
    return true;
}

static bool verify_pack_index_rejections(moe_pager & pager, const char * experts_bin, const char * index_txt) {
    std::ifstream input(index_txt);
    if (!input) return false;
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) lines.push_back(line);
    if (input.bad()) return false;

    std::vector<size_t> data_rows;
    for (size_t i = 0; i < lines.size(); ++i) {
        const size_t first = lines[i].find_first_not_of(" \t\r");
        if (first != std::string::npos && lines[i][first] != '#') data_rows.push_back(i);
    }
    if (data_rows.size() < 2) return false;

    auto join_lines = [](const std::vector<std::string> & source) {
        std::string result;
        for (const auto & item : source) result += item + "\n";
        return result;
    };
    auto reject_index = [&](const char * label, const std::vector<std::string> & bad_lines) {
        std::string temporary;
        if (!write_temp_index(join_lines(bad_lines), temporary)) return false;
        std::string error;
        const bool rejected = !pager.use_pack(experts_bin, temporary.c_str(), error) &&
                              !pager.pack_mode() && pager.pack_fd() < 0;
        unlink(temporary.c_str());
        printf("  malformed-index %-18s %s (%s)\n", label, rejected ? "PASS" : "FAIL", error.c_str());
        return rejected;
    };

    bool ok = true;
    std::vector<std::string> duplicate = lines;
    duplicate.push_back(lines[data_rows.front()]);
    ok &= reject_index("duplicate-layer", duplicate);

    std::vector<std::string> missing = lines;
    missing.erase(missing.begin() + (ptrdiff_t) data_rows.back());
    ok &= reject_index("missing-layer", missing);

    std::vector<std::string> bad_type = lines;
    {
        std::istringstream row(bad_type[data_rows.front()]);
        int layer, gu, dt;
        int64_t base, blob;
        uint64_t go, uo, doff;
        if (!(row >> layer >> gu >> dt >> base >> blob >> go >> uo >> doff)) return false;
        std::ostringstream replacement;
        replacement << layer << " " << (gu + 1) << " " << dt << " " << base << " " << blob
                    << " " << go << " " << uo << " " << doff;
        bad_type[data_rows.front()] = replacement.str();
    }
    ok &= reject_index("wrong-tensor-type", bad_type);

    std::vector<std::string> bad_range = lines;
    {
        std::istringstream row(bad_range[data_rows.front()]);
        int layer, gu, dt;
        int64_t base, blob;
        uint64_t go, uo, doff;
        if (!(row >> layer >> gu >> dt >> base >> blob >> go >> uo >> doff)) return false;
        std::ostringstream replacement;
        replacement << layer << " " << gu << " " << dt << " " << INT64_MAX << " " << blob
                    << " " << go << " " << uo << " " << doff;
        bad_range[data_rows.front()] = replacement.str();
    }
    ok &= reject_index("out-of-range", bad_range);

    struct stat source_stat;
    if (stat(experts_bin, &source_stat) != 0 || source_stat.st_size <= 1) return false;
    std::string pattern = std::string(experts_bin) + ".truncated-XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int truncated_fd = mkstemp(writable.data());
    if (truncated_fd < 0) return false;
    const bool truncated_created = ftruncate(truncated_fd, source_stat.st_size - 1) == 0;
    close(truncated_fd);
    const std::string truncated_path = writable.data();
    if (!truncated_created) { unlink(truncated_path.c_str()); return false; }
    std::string error;
    const bool truncated_rejected = !pager.use_pack(truncated_path.c_str(), index_txt, error) &&
                                    !pager.pack_mode() && pager.pack_fd() < 0;
    unlink(truncated_path.c_str());
    printf("  malformed-pack  truncated-file    %s (%s)\n",
           truncated_rejected ? "PASS" : "FAIL", error.c_str());
    ok &= truncated_rejected;
    return ok;
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : MODEL;
    const int NL = argc > 2 ? atoi(argv[2]) : 48;
    const int NE = argc > 3 ? atoi(argv[3]) : 512;
    const int NU = argc > 4 ? atoi(argv[4]) : 10;
    if (argc > 5 && argc != 8) {
        fprintf(stderr, "usage: %s [GGUF [layers experts used [experts.bin index.txt reference.gguf]]]\n", argv[0]);
        return 2;
    }

    moe_pager pager;
    moe_pager reference_pager;
    moe_pager * source_pager = &pager;
    std::string err;
    if (!pager.init(model_path, NL, NE, NU, err)) {
        fprintf(stderr, "FAIL: pager.init(%s): %s\n", model_path, err.c_str());
        return 1;
    }
    if (argc == 8) {
        if (!verify_pack_index_rejections(pager, argv[5], argv[6])) {
            fprintf(stderr, "FAIL: malformed pack/index rejection regression\n");
            return 1;
        }
        if (!pager.use_pack(argv[5], argv[6], err)) {
            fprintf(stderr, "FAIL: pager.use_pack(%s): %s\n", argv[5], err.c_str());
            return 1;
        }
        if (!reference_pager.init(argv[7], NL, NE, NU, err, 2)) {
            fprintf(stderr, "FAIL: reference pager.init(%s): %s\n", argv[7], err.c_str());
            return 1;
        }
        source_pager = &reference_pager;
        printf("pack mode: experts=%s index=%s reference=%s\n", argv[5], argv[6], argv[7]);
    }
    printf("pager: layers=%d experts=%d used=%d bytes/expert=%zu (%.3f MiB)\n",
           pager.n_layer(), pager.n_expert(), pager.n_used(),
           pager.bytes_per_expert(), pager.bytes_per_expert() / 1048576.0);

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { perror("open model"); return 1; }

    // Native expert sizes exclude direct-I/O alignment padding.
    const int probes[][2] = { {0,0}, {0,1}, {std::min(7, NL-1), std::min(123, NE-1)}, {std::min(23, NL-1), std::min(511, NE-1)}, {NL-1, NE-1} };
    int bad = 0;
    size_t biggest = 0;
    for (auto & pr : probes) {
        for (int r = 0; r < 3; r++) biggest = std::max(biggest, pager.expert_role_bytes(pr[0], r));
    }
    char *h_ref = (char *) malloc(biggest);
    char *h_pag = (char *) malloc(biggest);
    char *d_dst = nullptr;
    cudaMalloc(&d_dst, 3 * biggest);
    cudaMemset(d_dst, 0, 3 * biggest);
    for (auto & pr : probes) {
        const int il = pr[0], e = pr[1];
        const moe_expert_slice * sl = source_pager->slices(il, e);
        size_t want[3];
        for (int r = 0; r < 3; r++) want[r] = pager.expert_role_bytes(il, r);
        void *dg = d_dst, *du = d_dst + want[0], *dd = d_dst + want[0] + want[1];
        if (!pager.fetch(il, e, dg, want[0], du, want[1], dd, want[2])) {
            fprintf(stderr, "FAIL: fetch(%d,%d)\n", il, e); bad = 1; continue;
        }
        bool ok = true;
        const void *dsts[3] = { dg, du, dd };
        const char *roles[3] = { "gate", "up", "down" };
        for (int r = 0; r < 3 && ok; r++) {
            if (!read_reference(sl[r].fd >= 0 ? sl[r].fd : fd, sl[r], h_ref, want[r])) {
                fprintf(stderr, "FAIL: reference pread L%d E%d %s\n", il, e, roles[r]); bad = 1; ok = false; break;
            }
            cudaMemcpy(h_pag, dsts[r], want[r], cudaMemcpyDeviceToHost);
            if (memcmp(h_ref, h_pag, want[r]) != 0) ok = false;
        }
        printf("  fetch(L%2d,E%3d): compared %zu bytes  %s\n",
               il, e, want[0] + want[1] + want[2], ok ? "EXACT MATCH" : "*** MISMATCH ***");
        if (!ok) bad = 1;
    }
    // Async path: queue prefetches, then consume them via commit and verify
    // byte-exactness. This is the exact sequence the prefetch pipeline uses:
    // non-blocking reads issued ahead, demand commits that find them in flight.
    // Use per-layer pool geometry (gate/up/down byte sizes differ by layer).
    {
        const int aprobes[][2] = { {std::min(1, NL-1), std::min(19, NE-1)}, {std::min(8, NL-1), std::min(44, NE-1)}, {std::min(11, NL-1), std::min(300, NE-1)}, {std::min(37, NL-1), std::min(7, NE-1)} };
        // issue all prefetches first so reads overlap in the ring
        for (auto & pr : aprobes) {
            if (!pager.prefetch(pr[0], pr[1])) {
                fprintf(stderr, "FAIL: prefetch(%d,%d) refused\n", pr[0], pr[1]); bad = 1;
            }
        }
        for (auto & pr : aprobes) {
            const int il = pr[0], e = pr[1];
            const moe_expert_slice * sl = source_pager->slices(il, e);
            size_t want[3];
            for (int r = 0; r < 3; r++) want[r] = pager.expert_role_bytes(il, r);
            // destination sized generously; commit only writes want[] bytes
            static char *d_dst = nullptr;
            static size_t d_cap = 0;
            size_t need = want[0] + want[1] + want[2];
            if (need > d_cap) {
                if (d_dst) cudaFree(d_dst);
                cudaMalloc(&d_dst, need);
                d_cap = need;
            }
            void *dg = d_dst, *du = d_dst + want[0], *dd = d_dst + want[0] + want[1];
            if (!pager.commit(il, e, dg, want[0], du, want[1], dd, want[2])) {
                fprintf(stderr, "FAIL: commit(%d,%d)\n", il, e); bad = 1; continue;
            }
            bool ok = true;
            const void *dsts[3] = { dg, du, dd };
            for (int r = 0; r < 3 && ok; r++) {
                std::vector<char> ref(want[r]), got(want[r]);
                if (!read_reference(sl[r].fd >= 0 ? sl[r].fd : fd, sl[r], ref.data(), want[r])) {
                    fprintf(stderr, "FAIL: reference pread async L%d E%d r%d\n", il, e, r); bad = 1; ok = false; break;
                }
                cudaMemcpy(got.data(), dsts[r], want[r], cudaMemcpyDeviceToHost);
                if (memcmp(ref.data(), got.data(), want[r]) != 0) { ok = false; }
            }
            printf("  async(L%2d,E%3d): %zu bytes via prefetch+commit  %s (pf_hit=%ld)\n",
                   il, e, need, ok ? "EXACT MATCH" : "*** MISMATCH ***", pager.stats_pf_hit());
            if (!ok) bad = 1;
        }
    }
    // Caller-owned host copies must survive reuse of the pager staging arena.
    {
        const int layer = NL / 2;
        const int expert = NE / 3;
        const moe_expert_slice * slices = source_pager->slices(layer, expert);
        const size_t gate_bytes = pager.expert_role_bytes(layer, 0);
        const size_t up_bytes   = pager.expert_role_bytes(layer, 1);
        const size_t down_bytes = pager.expert_role_bytes(layer, 2);
        const size_t total_bytes = gate_bytes + up_bytes + down_bytes;

        char * device = nullptr;
        char * cache = nullptr;
        if (cudaMalloc(&device, total_bytes) != cudaSuccess ||
            cudaHostAlloc(&cache, total_bytes, cudaHostAllocDefault) != cudaSuccess) {
            bad = 1;
        } else {
            memset(cache, 0x5A, total_bytes);
            moe_batch_target bad_cap = {expert, device, gate_bytes, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes - 1};
            bool rejected_cap = !pager.fetch_batch(layer, &bad_cap, 1, nullptr);
            bool cap_intact = std::all_of(cache, cache + total_bytes, [](char byte) { return byte == 0x5A; });
            printf("  capacity-rejection: %s\n", (rejected_cap && cap_intact) ? "PASS" : "FAIL");
            if (!rejected_cap || !cap_intact) bad = 1;

            memset(cache, 0x3C, total_bytes);
            moe_batch_target bad_size = {expert, device, gate_bytes - 1, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes};
            bool rejected_size = !pager.fetch_batch(layer, &bad_size, 1, nullptr);
            bool size_intact = std::all_of(cache, cache + total_bytes, [](char byte) { return byte == 0x3C; });
            printf("  size-rejection: %s\n", (rejected_size && size_intact) ? "PASS" : "FAIL");
            if (!rejected_size || !size_intact) bad = 1;

            memset(cache, 0, total_bytes);
            moe_batch_target target = {expert, device, gate_bytes, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes};
            bool exact = pager.fetch_batch(layer, &target, 1, nullptr);
            size_t offset = 0;
            for (int role = 0; role < 3 && exact; ++role) {
                const size_t bytes = pager.expert_role_bytes(layer, role);
                std::vector<char> reference(bytes), actual(bytes);
                exact = read_reference(slices[role].fd, slices[role], reference.data(), bytes) &&
                        cudaMemcpy(actual.data(), device + offset, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        reference == actual &&
                        memcmp(reference.data(), cache + offset, bytes) == 0;
                offset += bytes;
            }
            printf("  write-allocate(L%2d,E%3d): %zu bytes in host and device %s\n",
                   layer, expert, total_bytes, exact ? "EXACT MATCH" : "MISMATCH");
            if (!exact) bad = 1;

            const int expert2 = (expert + 1) % NE;
            const moe_expert_slice * slices2 = source_pager->slices(layer, expert2);
            const size_t total_bytes2 = pager.expert_role_bytes(layer, 0) + pager.expert_role_bytes(layer, 1) + pager.expert_role_bytes(layer, 2);
            if (total_bytes2 <= total_bytes) {
                moe_batch_target target2 = {expert2, device, gate_bytes, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes};
                bool reuse_ok = pager.fetch_batch(layer, &target2, 1, nullptr);
                size_t off2 = 0;
                for (int role = 0; role < 3 && reuse_ok; ++role) {
                    const size_t bytes = pager.expert_role_bytes(layer, role);
                    std::vector<char> reference(bytes), actual(bytes);
                    reuse_ok = read_reference(slices2[role].fd, slices2[role], reference.data(), bytes) &&
                               cudaMemcpy(actual.data(), device + off2, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                               reference == actual &&
                               memcmp(reference.data(), cache + off2, bytes) == 0;
                    off2 += bytes;
                }
                bool refetch_ok = reuse_ok && pager.fetch_batch(layer, &target, 1, nullptr);
                size_t off1 = 0;
                for (int role = 0; role < 3 && refetch_ok; ++role) {
                    const size_t bytes = pager.expert_role_bytes(layer, role);
                    std::vector<char> reference(bytes);
                    refetch_ok = read_reference(slices[role].fd, slices[role], reference.data(), bytes) &&
                                 memcmp(reference.data(), cache + off1, bytes) == 0;
                    off1 += bytes;
                }
                printf("  reuse-stability(L%2d,E%3d<->E%3d): %s\n", layer, expert, expert2, refetch_ok ? "PASS" : "FAIL");
                if (!refetch_ok) bad = 1;
            }

            // Consume an outstanding prefetch through the batched path.
            const int expert_pref = (expert + 5) % NE;
            if (pager.prefetch(layer, expert_pref)) {
                pager.reap();
                moe_batch_target target_pref = {expert_pref, device, gate_bytes, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes};
                const long pf_hit_before = pager.stats_pf_hit();
                bool pref_ok = pager.fetch_batch(layer, &target_pref, 1, nullptr);
                const moe_expert_slice * slices_pref = source_pager->slices(layer, expert_pref);
                size_t off_pref = 0;
                for (int role = 0; role < 3 && pref_ok; ++role) {
                    const size_t bytes = pager.expert_role_bytes(layer, role);
                    std::vector<char> reference(bytes);
                    pref_ok = read_reference(slices_pref[role].fd, slices_pref[role], reference.data(), bytes) &&
                              memcmp(reference.data(), cache + off_pref, bytes) == 0;
                    off_pref += bytes;
                }
                printf("  prefetch-batch(L%2d,E%3d): %s (pf_hit=%ld->%ld)\n",
                       layer, expert_pref, pref_ok ? "PASS" : "FAIL", pf_hit_before, pager.stats_pf_hit());
                if (!pref_ok) bad = 1;
            }

            printf("  telemetry: disk_wait_us=%ld dma_ms=%.3f dma_wait_us=%ld dma_bytes=%ld\n",
                   (long) pager.stats_batch_disk_wait_us(), pager.stats_batch_h2d_ms(),
                   (long) pager.stats_dma_wait_us(), (long) pager.stats_dma_bytes());
        }
        if (device) cudaFree(device);
        if (cache) cudaFreeHost(cache);
    }
    // Mix host-cache hits and pending prefetches, then reuse the staging arena.
    if (!pager.init_cache(pager.bytes_per_expert() * 4)) bad = 1;
    const int batch_layers[] = {0, NL / 2, NL - 1};
    for (int il : batch_layers) {
        const int count = std::min(36, NE);
        size_t want[3];
        for (int role = 0; role < 3; ++role) want[role] = pager.expert_role_bytes(il, role);
        const size_t stride = want[0] + want[1] + want[2];
        char * device = nullptr;
        if (cudaMalloc(&device, stride * (size_t) count) != cudaSuccess) { bad = 1; break; }
        if (!pager.commit(il, 0, device, want[0], device + want[0], want[1], device + want[0] + want[1], want[2])) bad = 1;
        if (count > 1 && !pager.prefetch(il, count - 1)) bad = 1;
        std::vector<moe_batch_target> targets((size_t) count);
        for (int e = 0; e < count; ++e) {
            char * destination = device + (size_t) e * stride;
            targets[(size_t) e] = {e, destination, want[0], destination + want[0], want[1], destination + want[0] + want[1], want[2]};
        }
        for (int pass = 0; pass < (NE > count ? 2 : 1); ++pass) {
            if (pass == 1) {
                for (int e = 0; e < count; ++e) targets[(size_t) e].expert = (e + count) % NE;
            }
            if (!pager.fetch_batch(il, targets.data(), count)) { bad = 1; break; }
            bool exact = true;
            for (int e = 0; e < count; ++e) {
                const moe_expert_slice * slices = source_pager->slices(il, targets[(size_t) e].expert);
                size_t offset = 0;
                for (int role = 0; role < 3; ++role) {
                    std::vector<char> reference(want[role]), actual(want[role]);
                    if (!read_reference(slices[role].fd, slices[role], reference.data(), want[role]) ||
                            cudaMemcpy(actual.data(), device + (size_t) e * stride + offset, want[role], cudaMemcpyDeviceToHost) != cudaSuccess || reference != actual) exact = false;
                    offset += want[role];
                }
            }
            printf("  batch(L%2d,N%2d,pass%d): %s\n", il, count, pass, exact ? "EXACT MATCH" : "MISMATCH");
            if (!exact) bad = 1;
        }
        cudaFree(device);
    }
    close(fd);
    free(h_ref); free(h_pag);
    cudaFree(d_dst);

    printf("\nresult: %s   io_ops=%ld io_bytes=%ld pf_hit=%ld disk_wait_us=%ld dma_ms=%.3f\n",
           bad ? "FAIL" : "PASS", pager.stats_io_ops(), pager.stats_io_bytes(), pager.stats_pf_hit(),
           (long) pager.stats_batch_disk_wait_us(), pager.stats_batch_h2d_ms());
    return bad;
}

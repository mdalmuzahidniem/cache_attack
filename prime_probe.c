#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

#define NUM_SETS         16384
#define CACHE_LINE_SIZE  64
#define NUM_REPEATS      100
#define WAIT_CYCLES      2000  // ~0.4 us at 4.9GHz

volatile uint8_t *hugepage;

// Simple cycle-wait
static inline void wait_cycles(uint64_t cycles) {
    uint64_t start = __rdtsc();
    while (__rdtsc() - start < cycles);
}

// Allocate large page memory
void allocate_hugepage() {
    hugepage = mmap(NULL, 2 * 1024 * 1024, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (hugepage == MAP_FAILED) {
        perror("mmap (hugepage) failed");
        exit(1);
    }
    printf("[+] Huge page allocated at %p\n", hugepage);
}

// Prime a given cache set
void prime_set(int set_index) {
    volatile uint8_t *addr = hugepage + (set_index * CACHE_LINE_SIZE);
    *addr;
}

// Probe a given cache set and measure latency
uint64_t probe_set(int set_index) {
    volatile uint8_t *addr = hugepage + (set_index * CACHE_LINE_SIZE);
    uint64_t start = __rdtsc();
    (void)*addr;
    uint64_t end = __rdtsc();
    return end - start;
}

int main() {
    allocate_hugepage();

    FILE *f = fopen("latency_log.csv", "w");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "Iteration,SetIndex,Latency\n");

    for (int repeat = 0; repeat < NUM_REPEATS; repeat++) {
        printf("[+] Scan iteration %d\n", repeat);

        // ---- PRIME all sets ----
        for (int set = 0; set < NUM_SETS; set++) {
            prime_set(set);
        }

        // ---- WAIT ----
        wait_cycles(WAIT_CYCLES);

        // ---- PROBE all sets ----
        for (int set = 0; set < NUM_SETS; set++) {
            uint64_t latency = probe_set(set);
            fprintf(f, "%d,%d,%lu\n", repeat, set, latency);
        }
    }

    fclose(f);
    printf("[+] Prime+Probe complete!\n");
    return 0;
}


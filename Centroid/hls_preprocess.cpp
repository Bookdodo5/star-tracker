#include <cstdint>
#include <cstring>
#include <ap_int.h>
#ifndef __SYNTHESIS__
#include <cstdio>
#endif

// Minimal HLS-friendly preprocessing pipeline.
// - Grayscale (RGB -> Y)
// - 3x3 box blur (fine)
// - 3x3 box blur repeated 8x (coarse)
// - DoG (fine - coarse)
// - Threshold (fixed)
// - Morphological open (3x3: erosion then dilation)

constexpr int MAX_WIDTH = 1024;
constexpr int MAX_HEIGHT = 1024;
constexpr int MAX_PIXELS = MAX_WIDTH * MAX_HEIGHT;

// Union-Find helpers for two-pass connected components
static int uf_find(int a, int *par) {
    int x = a;
    while (par[x] != x) x = par[x];
    int y = a;
    while (par[y] != x) { int tmp = par[y]; par[y] = x; y = tmp; }
    return x;
}

static void uf_union(int a, int b, int *par) {
    int ra = uf_find(a, par);
    int rb = uf_find(b, par);
    if (ra == rb) return;
    if (ra < rb) par[rb] = ra; else par[ra] = rb;
}

// 3x3 box blur using 3-line sliding buffers (HLS-friendly)
static void box_blur_3x3(const uint8_t *in, uint8_t *out, int width, int height) {
    #pragma HLS INLINE off
    static uint8_t line0[MAX_WIDTH];
    static uint8_t line1[MAX_WIDTH];
    static uint8_t line2[MAX_WIDTH];

    for (int y = 0; y <= height; ++y) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_HEIGHT+1
        // shift lines: line0 <- line1, line1 <- line2
        for (int x = 0; x < width; ++x) {
            #pragma HLS PIPELINE II=1
            line0[x] = line1[x];
            line1[x] = line2[x];
        }

        // read current row into line2
        if (y < height) {
            for (int x = 0; x < width; ++x) {
                #pragma HLS PIPELINE II=1
                line2[x] = in[y * width + x];
            }
        }

        int out_y = y - 1;
        if (out_y >= 0) {
            // compute blur for this row using line0,line1,line2
            for (int x = 0; x < width; ++x) {
                #pragma HLS PIPELINE II=1
                int sum = 0;
                int cnt = 0;
                // row -1
                if (out_y > 0) {
                    int nx = x - 1; if (nx >= 0) { sum += line0[nx]; ++cnt; }
                    { sum += line0[x]; ++cnt; }
                    nx = x + 1; if (nx < width) { sum += line0[nx]; ++cnt; }
                }
                // row 0
                {
                    int nx = x - 1; if (nx >= 0) { sum += line1[nx]; ++cnt; }
                    { sum += line1[x]; ++cnt; }
                    nx = x + 1; if (nx < width) { sum += line1[nx]; ++cnt; }
                }
                // row +1
                if (out_y + 1 < height) {
                    int nx = x - 1; if (nx >= 0) { sum += line2[nx]; ++cnt; }
                    { sum += line2[x]; ++cnt; }
                    nx = x + 1; if (nx < width) { sum += line2[nx]; ++cnt; }
                }

                out[out_y * width + x] = (uint8_t)(sum / (cnt ? cnt : 1));
            }
        }
    }
}

extern "C" {

void hls_preprocess(const uint8_t* rgb_in, uint8_t* dog_out, uint8_t* morph_out,
                   uint16_t* star_x, uint16_t* star_y, uint64_t* star_brightness, int* star_count,
                   int width, int height) {
    #pragma HLS INTERFACE m_axi port=rgb_in  offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=dog_out  offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=morph_out offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=rgb_in  bundle=control
    #pragma HLS INTERFACE s_axilite port=dog_out  bundle=control
    #pragma HLS INTERFACE s_axilite port=morph_out bundle=control
    #pragma HLS INTERFACE s_axilite port=width    bundle=control
    #pragma HLS INTERFACE s_axilite port=height   bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control
    #pragma HLS INTERFACE m_axi port=star_x offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=star_y offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=star_brightness offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=star_x  bundle=control
    #pragma HLS INTERFACE s_axilite port=star_y  bundle=control
    #pragma HLS INTERFACE s_axilite port=star_brightness  bundle=control
    #pragma HLS INTERFACE s_axilite port=star_count bundle=control

    static uint8_t gray[MAX_PIXELS];
    static uint8_t buf1[MAX_PIXELS];
    static uint8_t buf2[MAX_PIXELS];
    static uint8_t thresh[MAX_PIXELS];

    const int pxCount = width * height;

    // RGB -> gray (Y) using integer weights (approx of 0.299/0.587/0.114)
    for (int i = 0; i < pxCount; ++i) {
        #pragma HLS PIPELINE II=1
        const int ri = i * 3;
        const int r = rgb_in[ri + 2];
        const int g = rgb_in[ri + 1];
        const int b = rgb_in[ri + 0];
        const int y = (r * 299 + g * 587 + b * 114 + 500) / 1000;
        gray[i] = (uint8_t) (y & 0xFF);
    }

#ifndef __SYNTHESIS__
    {
        FILE* f = fopen("hls_gray.pgm", "wb");
        if (f) {
            fprintf(f, "P5\n%d %d\n255\n", width, height);
            fwrite(gray, 1, pxCount, f);
            fclose(f);
        }
    }
#endif

    // 3x3 box blur: buf1 = blur(gray) using sliding line-buffer
    box_blur_3x3(gray, buf1, width, height);

    // Coarse blur: apply box blur 8 times starting from gray -> buf2
    // We'll iteratively use buf2 and buf1 as ping-pong buffers.
    // Initialize buf2 = gray
    for (int i = 0; i < pxCount; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_PIXELS
        buf2[i] = gray[i];
    }

    // Coarse blur: apply box blur multiple times using line-buffer implementation
    const int COARSE_ITERS = 12;
    for (int iter = 0; iter < COARSE_ITERS; ++iter) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=COARSE_ITERS
        box_blur_3x3(buf2, buf1, width, height);
        for (int i = 0; i < pxCount; ++i) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_PIXELS
            buf2[i] = buf1[i];
        }
    }

#ifndef __SYNTHESIS__
    {
        FILE* f = fopen("hls_coarse.pgm", "wb");
        if (f) {
            fprintf(f, "P5\n%d %d\n255\n", width, height);
            fwrite(buf2, 1, pxCount, f);
            fclose(f);
        }
    }
#endif

    // Now buf1 = blur(gray) (fine) and buf2 = blurCoarse (after 8 iterations)
    // Note: buf1 currently contains result of last coarse iteration; recompute fine blur into buf1_fine
    static uint8_t buf_fine[MAX_PIXELS];
    // Recompute fine blur using sliding line-buffer
    box_blur_3x3(gray, buf_fine, width, height);

#ifndef __SYNTHESIS__
    {
        FILE* f = fopen("hls_fine.pgm", "wb");
        if (f) {
            fprintf(f, "P5\n%d %d\n255\n", width, height);
            fwrite(buf_fine, 1, pxCount, f);
            fclose(f);
        }
    }
#endif

    // DoG: dog_out = clamp(fine - coarse)
    for (int i = 0; i < pxCount; ++i) {
        #pragma HLS PIPELINE II=1
        int val = (int)buf_fine[i] - (int)buf2[i];
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        dog_out[i] = (uint8_t)val;
    }

    // Threshold (fixed at 20)
    const uint8_t threshVal = 20;
    for (int i = 0; i < pxCount; ++i) {
        #pragma HLS PIPELINE II=1
        thresh[i] = (dog_out[i] >= threshVal) ? 255 : 0;
    }

#ifndef __SYNTHESIS__
    {
        FILE* f = fopen("hls_threshold.pgm", "wb");
        if (f) {
            fprintf(f, "P5\n%d %d\n255\n", width, height);
            fwrite(thresh, 1, pxCount, f);
            fclose(f);
        }
    }
#endif

    // Morphological open: erosion then dilation (3x3)
    // Erosion -> buf1
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            #pragma HLS PIPELINE II=1
            bool allSet = true;
            for (int dy = -1; dy <= 1 && allSet; ++dy) {
                int ny = y + dy;
                if (ny < 0 || ny >= height) { allSet = false; break; }
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= width) { allSet = false; break; }
                    if (thresh[ny * width + nx] == 0) { allSet = false; break; }
                }
            }
            buf1[y * width + x] = allSet ? 255 : 0;
        }
    }

    // Dilation -> morph_out
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            #pragma HLS PIPELINE II=1
            bool anySet = false;
            for (int dy = -1; dy <= 1 && !anySet; ++dy) {
                int ny = y + dy;
                if (ny < 0 || ny >= height) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    if (nx < 0 || nx >= width) continue;
                    if (buf1[ny * width + nx] != 0) { anySet = true; break; }
                }
            }
            morph_out[y * width + x] = anySet ? 255 : 0;
        }
    }

    // Connected components: two-pass connected-component labeling (8-connectivity)
    const int K = 10;
    static int labels[MAX_PIXELS];
    static int parent[MAX_PIXELS]; // union-find parent
    for (int i = 0; i < pxCount; ++i) {
        #pragma HLS PIPELINE II=1
        labels[i] = 0;
        parent[i] = i;
    }

        // use static uf_find/uf_union helper functions defined above

    // First pass: assign provisional labels and record equivalences
    int nextLabel = 1;
    for (int y0 = 0; y0 < height; ++y0) {
        for (int x0 = 0; x0 < width; ++x0) {
            #pragma HLS PIPELINE II=1
            int idx = y0 * width + x0;
            if (morph_out[idx] == 0) { labels[idx] = 0; continue; }

            // check 4 neighbors (left, up-left, up, up-right) for 8-connectivity
            int neighbor_labels[4];
            int ncount = 0;
            if (x0 > 0) { int l = labels[idx - 1]; if (l) neighbor_labels[ncount++] = l; }
            if (x0 > 0 && y0 > 0) { int ul = labels[idx - width - 1]; if (ul) neighbor_labels[ncount++] = ul; }
            if (y0 > 0) { int u = labels[idx - width]; if (u) neighbor_labels[ncount++] = u; }
            if (x0 + 1 < width && y0 > 0) { int ur = labels[idx - width + 1]; if (ur) neighbor_labels[ncount++] = ur; }

            if (ncount == 0) {
                labels[idx] = nextLabel;
                parent[nextLabel] = nextLabel;
                ++nextLabel;
            } else {
                int minl = neighbor_labels[0];
                for (int ni = 1; ni < ncount; ++ni) if (neighbor_labels[ni] < minl) minl = neighbor_labels[ni];
                labels[idx] = minl;
                for (int ni = 0; ni < ncount; ++ni) if (neighbor_labels[ni] != minl) uf_union(minl, neighbor_labels[ni], parent);
            }
        }
    }

    // Prepare aggregates per root label
    static uint64_t area_arr[MAX_PIXELS];
    static uint64_t sumX_arr[MAX_PIXELS];
    static uint64_t sumY_arr[MAX_PIXELS];
    static uint64_t bright_arr[MAX_PIXELS];
    for (int i = 0; i < nextLabel; ++i) {
        #pragma HLS PIPELINE II=1
        area_arr[i] = 0;
        sumX_arr[i] = 0;
        sumY_arr[i] = 0;
        bright_arr[i] = 0;
    }

    // Second pass: resolve labels and accumulate
    for (int y0 = 0; y0 < height; ++y0) {
        for (int x0 = 0; x0 < width; ++x0) {
            #pragma HLS PIPELINE II=1
            int idx = y0 * width + x0;
            int lab = labels[idx];
            if (lab == 0) continue;
            int root = uf_find(lab, parent);
            labels[idx] = root;
            area_arr[root] += 1;
            sumX_arr[root] += (uint64_t)x0;
            sumY_arr[root] += (uint64_t)y0;
            bright_arr[root] += (uint64_t)gray[idx];
        }
    }

    // top-K arrays
    uint16_t top_x[10];
    uint16_t top_y[10];
    uint64_t top_b[10];
    for (int i = 0; i < K; ++i) { top_x[i]=0; top_y[i]=0; top_b[i]=0; }

    // scan roots and pick top-K by brightness with area filter
    for (int lbl = 1; lbl < nextLabel; ++lbl) {
        #pragma HLS PIPELINE II=1
        uint64_t area = area_arr[lbl];
        if (area < 10 || area > 500) continue;
        uint64_t brightness = bright_arr[lbl];
        int centerX = (int)(sumX_arr[lbl] / area);
        int centerY = (int)(sumY_arr[lbl] / area);

        for (int i = 0; i < K; ++i) {
            if (brightness > top_b[i]) {
                for (int j = K-1; j > i; --j) {
                    top_b[j] = top_b[j-1];
                    top_x[j] = top_x[j-1];
                    top_y[j] = top_y[j-1];
                }
                top_b[i] = brightness;
                top_x[i] = (uint16_t)centerX;
                top_y[i] = (uint16_t)centerY;
                break;
            }
        }
    }

#undef UF_FIND

    // write outputs
    int found = 0;
    for (int i = 0; i < K; ++i) {
        if (top_b[i] == 0) break;
        star_x[i] = top_x[i];
        star_y[i] = top_y[i];
        star_brightness[i] = top_b[i];
        ++found;
    }
    *star_count = found;
}

} // extern "C"

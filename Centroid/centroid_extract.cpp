#include <cstdint>
#include <cstring>
#include <cstdio>
#include <chrono>

// Centroid extraction pipeline:
// - Grayscale (RGB -> Y)
// - Box blur fine (3x3) and coarse (11x11) via integral image — one O(N) build, two O(1)/pixel queries
// - DoG (fine - coarse)
// - Threshold (fixed at 20)
// - Morphological open (3x3 separable: two 1-D erosion passes then two 1-D dilation passes)
// - Connected-component centroiding, top-K by brightness

// Sized for the largest frame we feed in (1920x1080 sim renders). The static
// buffers below are indexed by width*height, so the only hard requirement is
// width*height <= MAX_PIXELS. ponytail: bump these if a bigger image appears.
constexpr int MAX_WIDTH  = 1920;
constexpr int MAX_HEIGHT = 1080;
constexpr int MAX_PIXELS = MAX_WIDTH * MAX_HEIGHT;

// ── Union-Find (two-pass CCL) ─────────────────────────────────────────────────
static int uf_find(int a, int *par) {
    int x = a;
    while (par[x] != x) x = par[x];
    while (par[a] != x) { int t = par[a]; par[a] = x; a = t; }
    return x;
}
static void uf_union(int a, int b, int *par) {
    int ra = uf_find(a, par), rb = uf_find(b, par);
    if (ra != rb) { if (ra < rb) par[rb] = ra; else par[ra] = rb; }
}

// ── Integral image ─────────────────────────────────────────────────────────────
// Builds a 2-D prefix sum so any rectangular box mean is O(1) per pixel.
// uint32 is sufficient: 1024×1024×255 = 267M < 4G.
static void build_integral(const uint8_t *gray, uint32_t *integ, int width, int height) {
    for (int y = 0; y < height; ++y) {
        uint32_t row_sum = 0;
        for (int x = 0; x < width; ++x) {
            row_sum += gray[y * width + x];
            integ[y * width + x] = row_sum + (y > 0 ? integ[(y-1)*width + x] : 0);
        }
    }
}

// Queries the mean of a (2r+1)×(2r+1) box centred at (cx,cy), clamped to image bounds.
static inline uint8_t integral_box_mean(const uint32_t *integ, int width, int height,
                                         int cx, int cy, int r) {
    int y0 = cy - r; if (y0 < 0) y0 = 0;
    int y1 = cy + r; if (y1 >= height) y1 = height - 1;
    int x0 = cx - r; if (x0 < 0) x0 = 0;
    int x1 = cx + r; if (x1 >= width) x1 = width - 1;

    uint32_t sum = integ[y1*width + x1];
    if (y0 > 0) sum -= integ[(y0-1)*width + x1];
    if (x0 > 0) sum -= integ[y1*width + (x0-1)];
    if (y0 > 0 && x0 > 0) sum += integ[(y0-1)*width + (x0-1)];

    uint32_t area = (uint32_t)(y1 - y0 + 1) * (uint32_t)(x1 - x0 + 1);
    return (uint8_t)(sum / area);
}

// ── Separable morphological open (3×3 square SE) ──────────────────────────────
// Erosion  = min (binary AND) across 3-neighbour windows, row then column.
// Dilation = max (binary OR)  across 3-neighbour windows, row then column.
// Four linear passes instead of one 9-neighbour loop → ~5× fewer memory accesses.
static void morph_open_3x3(const uint8_t *in, uint8_t *out, int width, int height) {
    static uint8_t re[MAX_PIXELS];  // after row erosion
    static uint8_t ce[MAX_PIXELS];  // after column erosion (= full erosion)
    static uint8_t rd[MAX_PIXELS];  // after row dilation

    // Row erosion
    for (int y = 0; y < height; ++y) {
        const uint8_t *s = in + y * width;
        uint8_t       *d = re + y * width;
        d[0] = s[0] < s[1] ? s[0] : s[1];
        for (int x = 1; x < width - 1; ++x) {
            uint8_t m = s[x-1] < s[x] ? s[x-1] : s[x];
            d[x] = m < s[x+1] ? m : s[x+1];
        }
        d[width-1] = s[width-2] < s[width-1] ? s[width-2] : s[width-1];
    }

    // Column erosion
    for (int x = 0; x < width; ++x) {
        ce[x] = re[x] < re[width+x] ? re[x] : re[width+x];
        for (int y = 1; y < height-1; ++y) {
            uint8_t a = re[(y-1)*width+x], b = re[y*width+x], c = re[(y+1)*width+x];
            uint8_t m = a < b ? a : b;
            ce[y*width+x] = m < c ? m : c;
        }
        { int y = height-1; ce[y*width+x] = re[(y-1)*width+x] < re[y*width+x] ? re[(y-1)*width+x] : re[y*width+x]; }
    }

    // Row dilation
    for (int y = 0; y < height; ++y) {
        const uint8_t *s = ce + y * width;
        uint8_t       *d = rd + y * width;
        d[0] = s[0] > s[1] ? s[0] : s[1];
        for (int x = 1; x < width - 1; ++x) {
            uint8_t m = s[x-1] > s[x] ? s[x-1] : s[x];
            d[x] = m > s[x+1] ? m : s[x+1];
        }
        d[width-1] = s[width-2] > s[width-1] ? s[width-2] : s[width-1];
    }

    // Column dilation → out
    for (int x = 0; x < width; ++x) {
        out[x] = rd[x] > rd[width+x] ? rd[x] : rd[width+x];
        for (int y = 1; y < height-1; ++y) {
            uint8_t a = rd[(y-1)*width+x], b = rd[y*width+x], c = rd[(y+1)*width+x];
            uint8_t m = a > b ? a : b;
            out[y*width+x] = m > c ? m : c;
        }
        { int y = height-1; out[y*width+x] = rd[(y-1)*width+x] > rd[y*width+x] ? rd[(y-1)*width+x] : rd[y*width+x]; }
    }
}

extern "C" {

void extract_centroids(const uint8_t* rgb_in, uint8_t* dog_out, uint8_t* morph_out,
                   uint16_t* star_x, uint16_t* star_y, uint64_t* star_brightness, int* star_count,
                   int width, int height) {

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    static uint8_t  gray[MAX_PIXELS];
    static uint32_t integ[MAX_PIXELS];   // integral image of gray
    static uint8_t  fine[MAX_PIXELS];
    static uint8_t  coarse[MAX_PIXELS];
    static uint8_t  thresh[MAX_PIXELS];

    const int pxCount = width * height;

    // Guard the static-buffer ceiling: silently overflowing produced garbage
    // centroids (10 noise blobs that match no catalog tetrad). Fail loud instead.
    if (pxCount > MAX_PIXELS) {
        std::fprintf(stderr,
            "centroid: image %dx%d (%d px) exceeds MAX_PIXELS=%d; raise MAX_WIDTH/MAX_HEIGHT\n",
            width, height, pxCount, MAX_PIXELS);
        *star_count = 0;
        return;
    }

    // ── Grayscale ──────────────────────────────────────────────────────────────
    for (int i = 0; i < pxCount; ++i) {
        int r = rgb_in[i*3+2], g = rgb_in[i*3+1], b = rgb_in[i*3+0];
        gray[i] = (uint8_t)((r*299 + g*587 + b*114 + 500) / 1000);
    }
    auto t1 = clk::now();

    // ── Integral image (built once, queried twice) ─────────────────────────────
    build_integral(gray, integ, width, height);
    auto t2 = clk::now();

    // ── Fine blur: 3×3 box (radius=1) ─────────────────────────────────────────
    // ── Coarse blur: 11×11 box (radius=5) ≈ 12 iterations of 3×3 blur ────────
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            fine  [y*width+x] = integral_box_mean(integ, width, height, x, y, 1);
            coarse[y*width+x] = integral_box_mean(integ, width, height, x, y, 5);
        }
    }
    auto t3 = clk::now();

    // ── DoG + threshold ────────────────────────────────────────────────────────
    for (int i = 0; i < pxCount; ++i) {
        int v = (int)fine[i] - (int)coarse[i];
        dog_out[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        thresh[i]  = (dog_out[i] >= 20) ? 255 : 0;
    }
    auto t4 = clk::now();

    // ── Morphological open ────────────────────────────────────────────────────
    morph_open_3x3(thresh, morph_out, width, height);
    auto t5 = clk::now();

    // ── Connected components ──────────────────────────────────────────────────
    const int K = 20;
    static int labels[MAX_PIXELS];
    static int parent[MAX_PIXELS];
    for (int i = 0; i < pxCount; ++i) { labels[i] = 0; parent[i] = i; }

    int nextLabel = 1;
    for (int y0 = 0; y0 < height; ++y0) {
        for (int x0 = 0; x0 < width; ++x0) {
            int idx = y0*width + x0;
            if (!morph_out[idx]) { labels[idx] = 0; continue; }
            int nl[4], nc = 0;
            if (x0 > 0)              { int l = labels[idx-1];         if (l) nl[nc++]=l; }
            if (x0 > 0 && y0 > 0)   { int l = labels[idx-width-1];   if (l) nl[nc++]=l; }
            if (y0 > 0)              { int l = labels[idx-width];     if (l) nl[nc++]=l; }
            if (x0+1<width && y0>0)  { int l = labels[idx-width+1];  if (l) nl[nc++]=l; }
            if (nc == 0) { labels[idx] = nextLabel; parent[nextLabel] = nextLabel; ++nextLabel; }
            else {
                int minl = nl[0];
                for (int ni = 1; ni < nc; ++ni) if (nl[ni] < minl) minl = nl[ni];
                labels[idx] = minl;
                for (int ni = 0; ni < nc; ++ni) if (nl[ni] != minl) uf_union(minl, nl[ni], parent);
            }
        }
    }

    static uint64_t area_a[MAX_PIXELS], sumX_a[MAX_PIXELS], sumY_a[MAX_PIXELS], bright_a[MAX_PIXELS];
    for (int i = 0; i < nextLabel; ++i) area_a[i]=sumX_a[i]=sumY_a[i]=bright_a[i]=0;

    for (int y0 = 0; y0 < height; ++y0) {
        for (int x0 = 0; x0 < width; ++x0) {
            int idx = y0*width+x0;
            int lab = labels[idx]; if (!lab) continue;
            int root = uf_find(lab, parent); labels[idx] = root;
            area_a[root]++; sumX_a[root]+=x0; sumY_a[root]+=y0; bright_a[root]+=gray[idx];
        }
    }

    uint16_t top_x[20]={}, top_y[20]={}; uint64_t top_b[20]={};
    for (int lbl = 1; lbl < nextLabel; ++lbl) {
        uint64_t area = area_a[lbl];
        if (area < 4 || area > 500) continue;
        uint64_t bri = bright_a[lbl];
        int cx = (int)(sumX_a[lbl]/area), cy = (int)(sumY_a[lbl]/area);
        for (int i = 0; i < K; ++i) {
            if (bri > top_b[i]) {
                for (int j=K-1;j>i;--j){top_b[j]=top_b[j-1];top_x[j]=top_x[j-1];top_y[j]=top_y[j-1];}
                top_b[i]=bri; top_x[i]=(uint16_t)cx; top_y[i]=(uint16_t)cy; break;
            }
        }
    }
    auto t6 = clk::now();

    int found = 0;
    for (int i = 0; i < K; ++i) {
        if (!top_b[i]) break;
        star_x[i]=top_x[i]; star_y[i]=top_y[i]; star_brightness[i]=top_b[i]; ++found;
    }
    *star_count = found;

    auto us = [](auto a, auto b){ return (int)std::chrono::duration_cast<std::chrono::microseconds>(b-a).count(); };
    fprintf(stderr, "centroid timing (us): gray=%d integral=%d blurs=%d dog+thresh=%d morph=%d ccl=%d total=%d\n",
            us(t0,t1), us(t1,t2), us(t2,t3), us(t3,t4), us(t4,t5), us(t5,t6), us(t0,t6));
}

} // extern "C"

// CLI wrapper for the centroid extractor.
//
// Reconstructed entry point: the original centroid_cli.cpp was an untracked file
// lost when the build configuration was deleted. It reads a binary PPM (P6),
// runs extract_centroids(), and writes the index,x,y,brightness CSV that the
// identifier (read_centroid_csv in identify_from_centroids.c) consumes.
//
// Usage: centroid_extract <in.ppm> <out.csv> [morph_passes=1]
//   morph_passes: 0 for real night-sky frames (1-2 px stars), 1 for satellite
//   blobs (>=3x3), N for noisier sensors. See extract_centroids() comment.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void extract_centroids(const uint8_t* rgb_in, uint8_t* dog_out, uint8_t* morph_out,
                                  uint16_t* star_x, uint16_t* star_y, uint64_t* star_brightness,
                                  int* star_count, int width, int height, int morph_passes);

/** Reads one whitespace-delimited non-negative integer from a P6 header,
 *  skipping '#' comment lines. Returns -1 on EOF. */
static int read_ppm_token(FILE* file) {
    int character = fgetc(file);
    for (;;) {
        while (character == ' ' || character == '\t' || character == '\n' || character == '\r')
            character = fgetc(file);
        if (character == '#') {                 // comment runs to end of line
            while (character != '\n' && character != EOF) character = fgetc(file);
            continue;
        }
        break;
    }
    if (character == EOF) return -1;
    int value = 0;
    while (character >= '0' && character <= '9') { value = value * 10 + (character - '0'); character = fgetc(file); }
    return value;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in.ppm> <out.csv> [morph_passes=1]\n", argv[0]);
        return 1;
    }
    const int morph_passes = (argc > 3) ? atoi(argv[3]) : 1;

    FILE* image = fopen(argv[1], "rb");
    if (!image) { fprintf(stderr, "centroid: cannot open %s\n", argv[1]); return 1; }

    char magic[3] = {0};
    if (fscanf(image, "%2s", magic) != 1 || magic[0] != 'P' || magic[1] != '6') {
        fprintf(stderr, "centroid: %s is not a binary PPM (P6)\n", argv[1]);
        fclose(image); return 1;
    }
    int width = read_ppm_token(image);
    int height = read_ppm_token(image);
    int maxval = read_ppm_token(image);
    if (width <= 0 || height <= 0 || maxval != 255) {
        fprintf(stderr, "centroid: bad PPM header (w=%d h=%d max=%d; need 8-bit)\n", width, height, maxval);
        fclose(image); return 1;
    }
    // read_ppm_token already consumed the single whitespace after maxval, so the
    // file is now positioned at the first pixel byte — do not consume another.
    const size_t pixels = (size_t)width * height;
    std::vector<uint8_t> rgb(pixels * 3);
    if (fread(rgb.data(), 1, rgb.size(), image) != rgb.size()) {
        fprintf(stderr, "centroid: truncated pixel data in %s\n", argv[1]);
        fclose(image); return 1;
    }
    fclose(image);

    std::vector<uint8_t> dog(pixels), morph(pixels);
    uint16_t star_x[20], star_y[20];
    uint64_t star_brightness[20];
    int star_count = 0;
    extract_centroids(rgb.data(), dog.data(), morph.data(),
                      star_x, star_y, star_brightness, &star_count,
                      width, height, morph_passes);

    FILE* csv = fopen(argv[2], "w");
    if (!csv) { fprintf(stderr, "centroid: cannot write %s\n", argv[2]); return 1; }
    fprintf(csv, "index,x,y,brightness\n");
    for (int i = 0; i < star_count; ++i)
        fprintf(csv, "%d,%u,%u,%llu\n", i, star_x[i], star_y[i], (unsigned long long)star_brightness[i]);
    fclose(csv);

    fprintf(stderr, "centroid: wrote %d stars to %s\n", star_count, argv[2]);
    return 0;
}

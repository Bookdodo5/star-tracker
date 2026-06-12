#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <cctype>

extern "C" void hls_preprocess(const uint8_t* rgb_in, uint8_t* dog_out, uint8_t* morph_out,
                                uint16_t* star_x, uint16_t* star_y, uint64_t* star_brightness, int* star_count,
                                int width, int height);

/**
 * Reads one PPM/PGM header token while skipping whitespace and comments.
 */
static bool readToken(std::istream& input, std::string& token) {
    token.clear();
    char ch = 0;
    while (input.get(ch)) {
        if (ch == '#') { input.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); continue; }
        if (!isspace((unsigned char)ch)) { token.push_back(ch); break; }
    }
    if (token.empty()) return false;
    while (input.get(ch)) {
        if (isspace((unsigned char)ch)) break;
        token.push_back(ch);
    }
    return true;
}

/**
 * Loads a binary P6 PPM or P5 PGM image as RGB bytes for the HLS centroid pipeline.
 */
static bool loadPpmOrPgm(const std::string& path, std::vector<uint8_t>& rgb, int& width, int& height) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::string magic;
    if (!readToken(file, magic)) return false;
    if (magic != "P6" && magic != "P5") return false;
    std::string token;
    if (!readToken(file, token)) return false; width = std::stoi(token);
    if (!readToken(file, token)) return false; height = std::stoi(token);
    if (!readToken(file, token)) return false; int maxv = std::stoi(token);
    if (maxv != 255) return false;
    const size_t px = (size_t)width * (size_t)height;
    if (magic == "P6") {
        rgb.resize(px * 3);
        file.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
        return static_cast<size_t>(file.gcount()) == rgb.size();
    }
    // P5
    std::vector<uint8_t> gray(px);
    file.read(reinterpret_cast<char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    if (static_cast<size_t>(file.gcount()) != gray.size()) return false;
    rgb.resize(px * 3);
    for (size_t i = 0; i < px; ++i) {
        rgb[i*3 + 0] = gray[i];
        rgb[i*3 + 1] = gray[i];
        rgb[i*3 + 2] = gray[i];
    }
    return true;
}

/**
 * Writes an 8-bit grayscale debug image as binary P5 PGM.
 */
static bool savePgm(const std::string& path, const std::vector<uint8_t>& img, int width, int height) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file << "P5\n" << width << ' ' << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
    return static_cast<bool>(file);
}

/**
 * Writes HLS centroid output in the CSV format consumed by the C star identifier.
 */
static bool saveStarsCsv(
    const std::string& path,
    const uint16_t *star_x,
    const uint16_t *star_y,
    const uint64_t *star_brightness,
    int star_count
) {
    std::ofstream file(path);
    if (!file) return false;
    file << "index,x,y,brightness\n";
    for (int starIndex = 0; starIndex < star_count; ++starIndex) {
        file << (starIndex + 1) << ','
             << star_x[starIndex] << ','
             << star_y[starIndex] << ','
             << star_brightness[starIndex] << '\n';
    }
    return static_cast<bool>(file);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: test_run_with_file <input.ppm> [output_stars.csv]\n"; return 1; }
    std::string inputImagePath = argv[1];
    std::string outputStarsCsvPath = (argc >= 3) ? argv[2] : "stars.csv";
    int width=0, height=0;
    std::vector<uint8_t> rgb;
    if (!loadPpmOrPgm(inputImagePath, rgb, width, height)) { std::cerr << "Failed to load " << inputImagePath << '\n'; return 1; }
    const size_t px = (size_t)width * (size_t)height;
    std::vector<uint8_t> dog(px);
    std::vector<uint8_t> morph(px);

    uint16_t star_x[10];
    uint16_t star_y[10];
    uint64_t star_brightness[10];
    int star_count = 0;
    hls_preprocess(rgb.data(), dog.data(), morph.data(), star_x, star_y, star_brightness, &star_count, width, height);

    if (!savePgm("dog.pgm", dog, width, height)) std::cerr << "Failed to write dog.pgm\n";
    if (!savePgm("morph.pgm", morph, width, height)) std::cerr << "Failed to write morph.pgm\n";
    if (!saveStarsCsv(outputStarsCsvPath, star_x, star_y, star_brightness, star_count)) { std::cerr << "Failed to write " << outputStarsCsvPath << '\n'; return 1; }

    int dog_nonzero=0,morph_nonzero=0; uint64_t dog_sum=0,morph_sum=0;
    for (size_t i=0;i<px;++i) { if (dog[i]) { ++dog_nonzero; dog_sum+=dog[i]; } if (morph[i]) { ++morph_nonzero; morph_sum+=morph[i]; } }
    std::cout<<"dog_nonzero="<<dog_nonzero<<" dog_avg="<<(dog_nonzero? (dog_sum/dog_nonzero):0)<<"\n";
    std::cout<<"morph_nonzero="<<morph_nonzero<<" morph_avg="<<(morph_nonzero? (morph_sum/morph_nonzero):0)<<"\n";
    std::cout<<"Wrote centroid CSV: "<<outputStarsCsvPath<<"\n";
    std::cout<<"Top "<<star_count<<" stars (x,y,brightness):\n";
    for (int i=0;i<star_count;++i) {
        std::cout<<i+1<<": "<<star_x[i]<<","<<star_y[i]<<" "<<star_brightness[i]<<"\n";
    }
    return 0;
}

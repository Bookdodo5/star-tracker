#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    std::size_t index(int x, int y) const {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
    }

    std::uint8_t& at(int x, int y) {
        return pixels[index(x, y)];
    }

    std::uint8_t at(int x, int y) const {
        return pixels[index(x, y)];
    }
};

struct PixelRef {
    std::uint8_t* data = nullptr;

    std::uint8_t& operator[](int channel) {
        return data[channel];
    }
};

struct ConstPixelRef {
    const std::uint8_t* data = nullptr;

    std::uint8_t operator[](int channel) const {
        return data[channel];
    }
};

struct ColorImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels; // Stored as B, G, R triplets.

    std::size_t index(int x, int y) const {
        return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 3;
    }

    PixelRef at(int x, int y) {
        return PixelRef{pixels.data() + index(x, y)};
    }

    ConstPixelRef at(int x, int y) const {
        return ConstPixelRef{pixels.data() + index(x, y)};
    }
};

struct Star {
    int x = 0;
    int y = 0;
    std::uint64_t brightness = 0;
};

static std::uint8_t clampToByte(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return static_cast<std::uint8_t>(value);
}

static std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

static bool readToken(std::istream& input, std::string& token) {
    token.clear();

    char ch = 0;
    while (input.get(ch)) {
        if (ch == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            token.push_back(ch);
            break;
        }
    }

    if (token.empty()) {
        return false;
    }

    while (input.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            break;
        }
        token.push_back(ch);
    }

    return true;
}

static bool loadPpmOrPgm(const std::string& path, ColorImage& image) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    std::string magic;
    if (!readToken(file, magic)) {
        return false;
    }

    if (magic != "P5" && magic != "P6") {
        return false;
    }

    std::string token;
    if (!readToken(file, token)) return false;
    image.width = std::stoi(token);
    if (!readToken(file, token)) return false;
    image.height = std::stoi(token);
    if (!readToken(file, token)) return false;
    const int maxValue = std::stoi(token);
    if (maxValue != 255 || image.width <= 0 || image.height <= 0) {
        return false;
    }

    const std::size_t pixelCount = static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height);
    image.pixels.resize(pixelCount * 3);

    if (magic == "P6") {
        file.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
        return static_cast<std::size_t>(file.gcount()) == image.pixels.size();
    }

    std::vector<std::uint8_t> gray(pixelCount);
    file.read(reinterpret_cast<char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    if (static_cast<std::size_t>(file.gcount()) != gray.size()) {
        return false;
    }

    for (std::size_t i = 0; i < pixelCount; ++i) {
        image.pixels[i * 3 + 0] = gray[i];
        image.pixels[i * 3 + 1] = gray[i];
        image.pixels[i * 3 + 2] = gray[i];
    }

    return true;
}

static bool savePgm(const std::string& path, const Image& image) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    file << "P5\n" << image.width << ' ' << image.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    return static_cast<bool>(file);
}

static bool savePpm(const std::string& path, const ColorImage& image) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    file << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    return static_cast<bool>(file);
}

static Image toGray(const ColorImage& input) {
    Image gray;
    gray.width = input.width;
    gray.height = input.height;
    gray.pixels.resize(static_cast<std::size_t>(gray.width) * static_cast<std::size_t>(gray.height));

    for (int y = 0; y < gray.height; ++y) {
        for (int x = 0; x < gray.width; ++x) {
            const auto rgb = input.at(x, y);
            const int value = (static_cast<int>(rgb[2]) * 299 + static_cast<int>(rgb[1]) * 587 + static_cast<int>(rgb[0]) * 114 + 500) / 1000;
            gray.at(x, y) = clampToByte(value);
        }
    }

    return gray;
}

static Image boxBlur3x3(const Image& input) {
    Image output;
    output.width = input.width;
    output.height = input.height;
    output.pixels.resize(static_cast<std::size_t>(output.width) * static_cast<std::size_t>(output.height));

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            int sum = 0;
            int count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= input.width || ny >= input.height) {
                        continue;
                    }
                    sum += input.at(nx, ny);
                    ++count;
                }
            }
            output.at(x, y) = static_cast<std::uint8_t>(sum / std::max(count, 1));
        }
    }

    return output;
}

static std::vector<std::uint8_t> doG(const Image& input) {
    const Image blurFine = boxBlur3x3(input);

    Image blurCoarse = input;
    for (int i = 0; i < 8; ++i) {
        blurCoarse = boxBlur3x3(blurCoarse);
    }

    std::vector<std::uint8_t> output(static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height));
    for (std::size_t i = 0; i < output.size(); ++i) {
        const int value = static_cast<int>(blurFine.pixels[i]) - static_cast<int>(blurCoarse.pixels[i]);
        output[i] = clampToByte(value);
    }
    return output;
}

static std::vector<std::uint8_t> thresholdImage(const std::vector<std::uint8_t>& input, std::uint8_t thresholdValue) {
    std::vector<std::uint8_t> output(input.size(), 0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] >= thresholdValue ? 255 : 0;
    }
    return output;
}

static std::vector<std::uint8_t> morphOpen3x3(const std::vector<std::uint8_t>& input, int width, int height) {
    std::vector<std::uint8_t> eroded(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
    std::vector<std::uint8_t> dilated(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool allSet = true;
            for (int dy = -1; dy <= 1 && allSet; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height || input[static_cast<std::size_t>(ny) * static_cast<std::size_t>(width) + static_cast<std::size_t>(nx)] == 0) {
                        allSet = false;
                        break;
                    }
                }
            }
            eroded[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] = allSet ? 255 : 0;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool anySet = false;
            for (int dy = -1; dy <= 1 && !anySet; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                        continue;
                    }
                    if (eroded[static_cast<std::size_t>(ny) * static_cast<std::size_t>(width) + static_cast<std::size_t>(nx)] != 0) {
                        anySet = true;
                        break;
                    }
                }
            }
            dilated[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] = anySet ? 255 : 0;
        }
    }

    return dilated;
}

static bool brighterFirst(const Star& lhs, const Star& rhs) {
    return lhs.brightness > rhs.brightness;
}

static std::vector<Star> findStars(const Image& gray, const std::vector<std::uint8_t>& thresholded, Image& labeledPreview) {
    const int width = gray.width;
    const int height = gray.height;
    const std::size_t pixelCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    labeledPreview = Image{width, height, thresholded};

    std::vector<int> labels(pixelCount, -1);
    std::vector<Star> stars;
    int currentLabel = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t startIndex = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
            if (thresholded[startIndex] == 0 || labels[startIndex] != -1) {
                continue;
            }

            std::vector<int> stackX(pixelCount);
            std::vector<int> stackY(pixelCount);
            int stackSize = 0;
            stackX[stackSize] = x;
            stackY[stackSize] = y;
            ++stackSize;
            labels[startIndex] = currentLabel;

            std::uint64_t area = 0;
            std::uint64_t sumX = 0;
            std::uint64_t sumY = 0;
            std::uint64_t brightness = 0;

            while (stackSize > 0) {
                --stackSize;
                const int cx = stackX[stackSize];
                const int cy = stackY[stackSize];

                ++area;
                sumX += static_cast<std::uint64_t>(cx);
                sumY += static_cast<std::uint64_t>(cy);
                brightness += gray.at(cx, cy);

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int nx = cx + dx;
                        const int ny = cy + dy;
                        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                            continue;
                        }
                        const std::size_t nIndex = static_cast<std::size_t>(ny) * static_cast<std::size_t>(width) + static_cast<std::size_t>(nx);
                        if (thresholded[nIndex] != 0 && labels[nIndex] == -1) {
                            labels[nIndex] = currentLabel;
                            stackX[stackSize] = nx;
                            stackY[stackSize] = ny;
                            ++stackSize;
                        }
                    }
                }
            }

            if (area >= 10 && area <= 500) {
                const int centerX = static_cast<int>(sumX / area);
                const int centerY = static_cast<int>(sumY / area);
                stars.push_back(Star{centerX, centerY, brightness});
            }

            ++currentLabel;
        }
    }

    std::sort(stars.begin(), stars.end(), brighterFirst);

    return stars;
}

static void drawCross(ColorImage& image, int x, int y, int halfSize) {
    for (int offset = -halfSize; offset <= halfSize; ++offset) {
        const int horizontalX = x + offset;
        const int verticalY = y + offset;

        if (horizontalX >= 0 && horizontalX < image.width && y >= 0 && y < image.height) {
            auto pixel = image.at(horizontalX, y);
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 255;
        }

        if (x >= 0 && x < image.width && verticalY >= 0 && verticalY < image.height) {
            auto pixel = image.at(x, verticalY);
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 255;
        }
    }
}

static void drawDigit(ColorImage& image, int x, int y, int digit) {
    static const std::uint8_t glyphs[10][5][3] = {
        {{1, 1, 1}, {1, 0, 1}, {1, 0, 1}, {1, 0, 1}, {1, 1, 1}},
        {{0, 1, 0}, {1, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 1, 1}},
        {{1, 1, 1}, {0, 0, 1}, {1, 1, 1}, {1, 0, 0}, {1, 1, 1}},
        {{1, 1, 1}, {0, 0, 1}, {1, 1, 1}, {0, 0, 1}, {1, 1, 1}},
        {{1, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 0, 1}, {0, 0, 1}},
        {{1, 1, 1}, {1, 0, 0}, {1, 1, 1}, {0, 0, 1}, {1, 1, 1}},
        {{1, 1, 1}, {1, 0, 0}, {1, 1, 1}, {1, 0, 1}, {1, 1, 1}},
        {{1, 1, 1}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}},
        {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}, {1, 0, 1}, {1, 1, 1}},
        {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}, {0, 0, 1}, {1, 1, 1}}
    };

    if (digit < 0 || digit > 9) {
        return;
    }

    const auto& glyph = glyphs[static_cast<std::size_t>(digit)];
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 3; ++col) {
            if (glyph[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] == 0) {
                continue;
            }
            const int px = x + col;
            const int py = y + row;
            if (px < 0 || py < 0 || px >= image.width || py >= image.height) {
                continue;
            }
            auto pixel = image.at(px, py);
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
        }
    }
}

static void drawNumber(ColorImage& image, int x, int y, int value) {
    std::string text = std::to_string(value);
    int cursorX = x;
    for (char ch : text) {
        drawDigit(image, cursorX, y, ch - '0');
        cursorX += 4;
    }
}

static ColorImage makePreview(const ColorImage& input, const std::vector<Star>& stars) {
    ColorImage output = input;
    for (std::size_t i = 0; i < stars.size(); ++i) {
        drawCross(output, stars[i].x, stars[i].y, 5);
        drawNumber(output, stars[i].x + 8, stars[i].y - 6, static_cast<int>(i + 1));
    }
    return output;
}

/**
 * Writes detected centroids in the CSV format consumed by the C star identifier.
 */
static bool saveStarsCsv(const std::string& path, const std::vector<Star>& stars) {
    std::ofstream file(path);
    if (!file) {
        return false;
    }
    file << "index,x,y,brightness\n";
    for (std::size_t starIndex = 0; starIndex < stars.size(); ++starIndex) {
        file << (starIndex + 1) << ','
             << stars[starIndex].x << ','
             << stars[starIndex].y << ','
             << stars[starIndex].brightness << '\n';
    }
    return static_cast<bool>(file);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: star_tracker <input.pgm|input.ppm>\n";
        return 1;
    }

    std::cout << "HLS friendly!" << std::endl;

    ColorImage input;
    if (!loadPpmOrPgm(argv[1], input)) {
        std::cerr << "Could not load image. This pure C++ version supports PGM/PPM only.\n";
        return 1;
    }

    const Image gray = toGray(input);
    const std::vector<std::uint8_t> dog = doG(gray);
    const std::vector<std::uint8_t> thresholded = morphOpen3x3(thresholdImage(dog, 20), gray.width, gray.height);

    Image thresholdPreview;
    const std::vector<Star> stars = findStars(gray, thresholded, thresholdPreview);
    const ColorImage preview = makePreview(input, stars);

    std::cout << "Detected stars: " << stars.size() << '\n';
    for (std::size_t i = 0; i < stars.size(); ++i) {
        std::cout << (i + 1) << ": x=" << stars[i].x << " y=" << stars[i].y << " brightness=" << stars[i].brightness << '\n';
    }

    const bool wroteGray = savePgm("gray.pgm", gray);
    const bool wroteDog = savePgm("dog.pgm", Image{gray.width, gray.height, dog});
    const bool wroteThreshold = savePgm("threshold.pgm", thresholdPreview);
    const bool wrotePreview = savePpm("visualized.ppm", preview);
    const bool wroteStars = saveStarsCsv("stars.csv", stars);

    int dog_nonzero=0, morph_nonzero=0;
    std::uint64_t dog_sum=0, morph_sum=0;
    for (std::size_t i=0; i<dog.size(); ++i) {
        if (dog[i]) { ++dog_nonzero; dog_sum+=dog[i]; }
        if (thresholded[i]) { ++morph_nonzero; morph_sum+=thresholded[i]; }
    }
    std::cout << "dog_nonzero=" << dog_nonzero << " dog_avg=" << (dog_nonzero ? dog_sum/dog_nonzero : 0) << "\n";
    std::cout << "morph_nonzero=" << morph_nonzero << " morph_avg=" << (morph_nonzero ? morph_sum/morph_nonzero : 0) << "\n";

    if (!wroteGray || !wroteDog || !wroteThreshold || !wrotePreview || !wroteStars) {
        std::cerr << "Warning: one or more output files could not be written.\n";
        return 1;
    }

    return 0;
}

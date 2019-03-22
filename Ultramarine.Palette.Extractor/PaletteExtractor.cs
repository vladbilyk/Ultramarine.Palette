using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace Ultramarine.Palette.Extractor
{
    // C# Port of Javascript version of palette extractor https://github.com/googleartsculture/art-palette
    public class PaletteExtractor
    {
        const int HISTOGRAM_SIZE_ = 4096;
        const double REF_X = 95.047;
        const int REF_Y = 100;
        const double REF_Z = 108.883;

        Dictionary<int, double[]> labs_ = new Dictionary<int, double[]>();
        List<int> weights_ = new List<int>();
        List<double[]> seeds_ = new List<double[]>();
        List<int> seedWeights_ = new List<int>();

        public List<Color> ProcessBitmap(Bitmap data, int paletteSize = 5)
        {
            ComputeHistogramFromImageData(data);
            SelectSeeds(paletteSize);
            ClusterColors();
            return ExportPalette();
        }

        public List<Color> ExportPalette()
        {
            return seeds_.Select(s => LabToColor(s)).ToList();
        }

        private static Color LabToColor(double[] lab)
        {
            var xyz = LabToXyz(lab);
            return XyzToColor(xyz);
        }

        private static double[] LabToXyz(double[] lab)
        {
            var p = (lab[0] + 16) / 116;
            return new[] {
              REF_X * Math.Pow(p + lab[1] / 500, 3),
              REF_Y * Math.Pow(p, 3),
              REF_Z * Math.Pow(p - lab[2] / 200, 3)
            };
        }

        private static Color XyzToColor(double[] xyz)
        {
            var x = xyz[0] / 100.0;
            var y = xyz[1] / 100.0;
            var z = xyz[2] / 100.0;
            var r = x * 3.2406 + y * -1.5372 + z * -0.4986;
            var g = x * -0.9689 + y * 1.8758 + z * 0.0415;
            var b = x * 0.0557 + y * -0.2040 + z * 1.0570;
            if (r > 0.0031308)
            {
                r = 1.055 * Math.Pow(r, 1 / 2.4) - .055;
            }
            else
            {
                r = 12.92 * r;
            }
            if (g > 0.0031308)
            {
                g = 1.055 * Math.Pow(g, 1 / 2.4) - .055;
            }
            else
            {
                g = 12.92 * g;
            }
            if (b > 0.0031308)
            {
                b = 1.055 * Math.Pow(b, 1 / 2.4) - .055;
            }
            else
            {
                b = 12.92 * b;
            }
            return Color.FromArgb(
                Math.Min(255, Math.Max(0, (int)Math.Round(r * 255))),
                Math.Min(255, Math.Max(0, (int)Math.Round(g * 255))),
                Math.Min(255, Math.Max(0, (int)Math.Round(b * 255))));
        }

        private void ClusterColors()
        {
            if (seeds_.Count == 0)
            {
                throw new Exception("Please select seeds before clustering");
            }

            var clusterIndices = Enumerable.Repeat(0, HISTOGRAM_SIZE_).ToList();
            seedWeights_ = new List<int>();
            var optimumReached = false;
            var i = 0;
            while (!optimumReached)
            {
                optimumReached = true;
                var newSeeds = new Dictionary<int, double[]>();
                seedWeights_ = Enumerable.Repeat(0, seeds_.Count).ToList();
                // Assign every bin of the color histogram to the closest seed.
                for (i = 0; i < HISTOGRAM_SIZE_; i++)
                {
                    if (weights_[i] == 0)
                    {
                        continue;
                    }

                    // Compute the index of the seed that is closest to the bin's color.
                    var clusterIndex = GetClosestSeedIndex(i);
                    // Optimum is reached when no cluster assignment changes.
                    if (optimumReached && clusterIndex != clusterIndices[i])
                    {
                        optimumReached = false;
                    }
                    // Assign bin to closest seed.
                    clusterIndices[i] = clusterIndex;
                    // Accumulate colors and weights per cluster.
                    AddColorToSeed(newSeeds, clusterIndex, i);
                }
                // Average accumulated colors to get new seeds.
                UpdateSeedsWithNewSeeds(newSeeds);
            }
        }

        private void UpdateSeedsWithNewSeeds(Dictionary<int, double[]> newSeeds)
        {
            for (var i = 0; i < seeds_.Count; i++)
            {
                if (!newSeeds.ContainsKey(i))
                {
                    newSeeds[i] = new[] { 0.0, 0, 0 };
                }

                if (seedWeights_[i] == 0)
                {
                    newSeeds[i] = new[] { 0.0, 0, 0 };
                }
                else
                {
                    var old = newSeeds[i];
                    var w = seedWeights_[i];
                    newSeeds[i] = new[] { old[0] / w, old[1] / w, old[2] / w };
                }

                // Update seeds.
                seeds_[i] = new[] { newSeeds[i][0], newSeeds[i][1], newSeeds[i][2] };
            }
        }

        private void AddColorToSeed(Dictionary<int, double[]> seeds, int clusterIndex, int histogramIndex)
        {
            if (!seeds.ContainsKey(clusterIndex))
            {
                seeds[clusterIndex] = new[] { 0.0, 0, 0 };
            }

            var old = seeds[clusterIndex];
            var l = labs_[histogramIndex];
            seeds[clusterIndex] = new[] { l[0] + old[0], l[1] + old[1], l[2] + old[2] };
            seedWeights_[clusterIndex] += weights_[histogramIndex];
        }

        private int GetClosestSeedIndex(int index)
        {
            var l = labs_[index];
            var w = weights_[index];
            var color = new[] { l[0] / w, l[1] / w, l[2] / w };
            var seedDistMin = double.MaxValue;
            var seedIndex = 0;
            for (var i = 0; i < seeds_.Count; i++)
            {
                var dist = DistanceSquared(seeds_[i], color);
                if (dist < seedDistMin)
                {
                    seedDistMin = dist;
                    seedIndex = i;
                }
            }
            return seedIndex;
        }

        private void SelectSeeds(int paletteSize)
        {
            // Reset histogram
            seeds_ = new List<double[]>();
            // Local copy of the weight bins to edit during seed selection.
            var mutableWeights = new List<double>(weights_.Select(w => (double)w));

            // Iteratively selects seeds as the heaviest bins in mutableWeights.
            // After selecting a seed, attenuates neighboring bin weights to increase
            // color variance.
            var maxIndex = 0;
            for (var i = 0; i < paletteSize; ++i)
            {
                // Get the index of the heaviest bin.
                maxIndex = GetHeaviestIndex(mutableWeights);

                // Check that the selected bin is not empty.
                // Otherwise, it means that the previous seeds already cover all
                // non-empty bins.
                if (mutableWeights[maxIndex] == 0)
                {
                    break;
                }

                // Set the new seed as the heaviest bin.
                var seedColor = AddSeedByIndex(maxIndex);

                // Force the next seed to be different (unless all bin weights are 0).
                mutableWeights[maxIndex] = 0;

                // Attenuate weights close to the seed to maximize distance between seeds.
                AttenuateWeightsAroundSeed(mutableWeights, seedColor);
            }
        }

        private void AttenuateWeightsAroundSeed(List<double> mutableWeights, double[] seedColor)
        {
            // For photos, we can use a higher coefficient, from 900 to 6400
            var squaredSeparationCoefficient = 3650;

            for (var i = 0; i < HISTOGRAM_SIZE_; i++)
            {
                var w = weights_[i];
                if (w > 0)
                {
                    var l = labs_[i];
                    var targetColor = new double[]
                    {
                        l[0] / w,
                        l[1] / w,
                        l[2] / w
                    };
                    mutableWeights[i] *= 1 - Math.Exp(-DistanceSquared(seedColor, targetColor) / squaredSeparationCoefficient);
                }
            }
        }

        static double DistanceSquared(double[] v1, double[] v2)
        {
            var x = v1[0] - v2[0];
            var y = v1[1] - v2[1];
            var z = v1[2] - v2[2];
            return x * x + y * y + z * z;
        }

        double[] AddSeedByIndex(int index)
        {
            var l = labs_[index];
            var w = weights_[index];
            var seedColor = new double[] {
                l[0] / w, l[1] / w, l[2] / w
            };
            seeds_.Add(seedColor);
            return seedColor;
        }

        static int GetHeaviestIndex(List<double> weights)
        {
            var heaviest = 0.0;
            var index = 0;
            for (var m = 0; m < HISTOGRAM_SIZE_; m++)
            {
                if (weights[m] > heaviest)
                {
                    heaviest = weights[m];
                    index = m;
                }
            }
            return index;
        }

        private void ComputeHistogramFromImageData(Bitmap data)
        {
            // reset histogram
            labs_ = new Dictionary<int, double[]>();
            weights_ = Enumerable.Repeat(0, HISTOGRAM_SIZE_).ToList();

            for (int i = 0; i < data.Width; ++i)
            {
                for (int j = 0; j < data.Height; ++j)
                {
                    var color = data.GetPixel(i, j);
                    var lab = ColorToLab(color);
                    var index = (int)((Math.Floor(color.R / 16.0) * 16 + Math.Floor(color.G / 16.0)) * 16 +
                        Math.Floor(color.B / 16.0));

                    if (!labs_.ContainsKey(index))
                    {
                        labs_[index] = lab;
                    }
                    else
                    {
                        var old = labs_[index];
                        labs_[index] = new[] { old[0] + lab[0], old[1] + lab[1], old[2] + lab[2] };
                    }

                    weights_[index]++;
                }
            }
        }

        static double[] ColorToLab(Color color)
        {
            var xyz = ColorToXyz(color);
            return XyzToLab(xyz[0], xyz[1], xyz[2]);
        }

        static double[] ColorToXyz(Color color)
        {
            var r = color.R / 255.0;
            var g = color.G / 255.0;
            var b = color.B / 255.0;

            if (r > 0.04045)
            {
                r = Math.Pow((r + .055) / 1.055, 2.4);
            }
            else
            {
                r = r / 12.92;
            }
            if (g > 0.04045)
            {
                g = Math.Pow((g + .055) / 1.055, 2.4);
            }
            else
            {
                g = g / 12.92;
            }
            if (b > 0.04045)
            {
                b = Math.Pow((b + .055) / 1.055, 2.4);
            }
            else
            {
                b = b / 12.92;
            }
            r = r * 100;
            g = g * 100;
            b = b * 100;
            return new[] {
              r * 0.4124 + g * 0.3576 + b * 0.1805,
              r * 0.2126 + g * 0.7152 + b * 0.0722, r * 0.0193 + g * 0.1192 + b * 0.9505
            };
        }

        static double[] XyzToLab(double x, double y, double z)
        {
            var xRatio = x / REF_X;
            var yRatio = y / REF_Y;
            var zRatio = z / REF_Z;
            return new[] {
              yRatio > 0.008856 ? 116 * Math.Pow(yRatio, 1.0 / 3) - 16 : 903.3 * yRatio,
              500 * (Transformation(xRatio) - Transformation(yRatio)),
              200 * (Transformation(yRatio) - Transformation(zRatio))
            };
        }

        static double Transformation(double t)
        {
            if (t > 0.008856)
            {
                return Math.Pow(t, 1.0 / 3);
            }
            return 7.787 * t + 16.0 / 116;
        }
    }
}

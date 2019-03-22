using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Ultramarine.Palette.Extractor;

namespace BulkImageProcessor
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Palette extractor");

            var files = new List<string> { @"D:\Livemaster\otkrytki\40908.jpg" };

            var palettes = await ExtractPalettesAsync(files);

            foreach (var palette in palettes)
            {
                Console.WriteLine($"File: {palette.FilePath} - Palette: {palette.Palette}");
            }
        }

        private static async Task<List<ImgPalette>> ExtractPalettesAsync(List<string> files)
        {
            var chunkCount = 0;
            var result = new List<ImgPalette>();

            foreach (var chunk in SplitList(files, 10))
            {
                chunkCount++;
                var tasks = chunk.Select(async filePath => await Task.Run(() =>
                {
                    try
                    {
                        using (var bitmap = new Bitmap(filePath))
                        {
                            var processor = new PaletteExtractor();
                            processor.ProcessBitmap(bitmap);
                            var palette = processor.ExportPalette();

                            var text = new StringBuilder();
                            for (int i = 0; i < palette.Count; ++i)
                            {
                                text.Append(palette[i].R.ToString("x2"));
                                text.Append(palette[i].G.ToString("x2"));
                                text.Append(palette[i].B.ToString("x2"));
                                if (i != palette.Count - 1)
                                {
                                    text.Append('-');
                                }
                            }
                            return new ImgPalette { Palette = text.ToString(), FilePath = filePath };
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine($"Error: {e.Message}. File: {filePath}");
                        return new ImgPalette { Palette = "error", FilePath = filePath };
                    }
                }));

                var processed = await Task.WhenAll(tasks);
                result.AddRange(processed);

                Console.WriteLine($"Processed: {chunkCount}");
            }

            return result;
        }


        public static IEnumerable<List<T>> SplitList<T>(List<T> locations, int nSize = 30)
        {
            for (int i = 0; i < locations.Count; i += nSize)
            {
                yield return locations.GetRange(i, Math.Min(nSize, locations.Count - i));
            }
        }


        class ImgPalette
        {
            public string Palette { get; set; }
            public string FilePath { get; set; }
        }

    }
}

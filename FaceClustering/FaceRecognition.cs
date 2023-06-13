using System.Collections.Generic;
using System;
using DlibDotNet;
using System.Net;

namespace FaceClustering
{
    public class FaceRecognition
    {
        private const string inputFilePath = "images/AdobeStock_238313033_Preview.jpeg";

        public static void RunFaceRecognition()
        {
            Console.WriteLine("Loading detectors...");

            using (var detector = Dlib.GetFrontalFaceDetector())
            using (var predictor = ShapePredictor.Deserialize("models/shape_predictor_5_face_landmarks.dat"))
            using (var dnn = DlibDotNet.Dnn.LossMetric.Deserialize("models/dlib_face_recognition_resnet_model_v1.dat"))
            using (var img = Dlib.LoadImage<RgbPixel>(inputFilePath))
            {
                var chips = new List<Matrix<RgbPixel>>();
                var faces = new List<Rectangle>();
                Console.WriteLine("Detecting faces...");
                foreach (var face in detector.Operator(img))
                {
                    var shape = predictor.Detect(img, face);
                    var faceChipDetail = Dlib.GetFaceChipDetails(shape, 150, 0.25);
                    var faceChip = Dlib.ExtractImageChip<RgbPixel>(img, faceChipDetail);
                    var matrix = new Matrix<RgbPixel>(faceChip);
                    chips.Add(matrix);
                    faces.Add(face);
                }
                Console.WriteLine($"   Found {chips.Count} faces in image");

                Console.WriteLine("Recognizing faces...");
                var descriptors = dnn.Operator(chips);

                var edges = new List<SamplePair>();
                for (uint i = 0; i < descriptors.Count; ++i)
                {
                    for (var j = i; j < descriptors.Count; ++j)
                    {
                        if (Dlib.Length(descriptors[i] - descriptors[j]) < 0.6)
                            edges.Add(new SamplePair(i, j));
                    }
                }

                Dlib.ChineseWhispers(edges, 100, out var clusters, out var labels);
                Console.WriteLine($"   Found {clusters} unique person(s) in the image");

                int numColors = 50;
                var palette = GenerateRandomPalette(numColors);

                for (var i = 0; i < faces.Count; i++)
                {
                    Dlib.DrawRectangle(img, faces[i], color: palette[labels[i]], thickness: 4);
                }

                var outputFolder = "output";
                Directory.CreateDirectory(outputFolder);
                Dlib.SaveJpeg(img, "output/output.jpg");
            }
        }

        private static RgbPixel[] GenerateRandomPalette(int numColors)
        {
            var palette = new RgbPixel[numColors];
            var random = new Random();

            for (int i = 0; i < numColors; i++)
            {
                byte r = (byte)random.Next(256);
                byte g = (byte)random.Next(256);
                byte b = (byte)random.Next(256);
                palette[i] = new RgbPixel(r, g, b);
            }

            return palette;
        }
    }
}

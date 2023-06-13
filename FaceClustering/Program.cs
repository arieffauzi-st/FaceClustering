using System.Collections.Generic;
using System;
using DlibDotNet;
using System.Net;

namespace FaceClustering
{
    /// <summary>
    /// The program class.
    /// </summary>
    class Program
    {
        // file paths
        private const string inputFilePath = "images/AdobeStock_238313033_Preview.jpeg";

        /// <summary>
        /// The program entry point.
        /// </summary>
        /// <param name="args">The command line arguments</param>
        static void Main(string[] args)
        {
            Console.WriteLine("Loading detectors...");

            // set up a face detector
            using (var detector = Dlib.GetFrontalFaceDetector())

            // set up a 5-point landmark detector
            using (var predictor = ShapePredictor.Deserialize("models/shape_predictor_5_face_landmarks.dat"))

            // set up a neural network for face recognition
            using (var dnn = DlibDotNet.Dnn.LossMetric.Deserialize("models/dlib_face_recognition_resnet_model_v1.dat"))

            // load the image
            using (var img = Dlib.LoadImage<RgbPixel>(inputFilePath))
            {
                // detect all faces
                var chips = new List<Matrix<RgbPixel>>();
                var faces = new List<Rectangle>();
                Console.WriteLine("Detecting faces...");
                foreach (var face in detector.Operator(img))
                {
                    // detect landmarks
                    var shape = predictor.Detect(img, face);

                    // extract normalized and rotated 150x150 face chip
                    var faceChipDetail = Dlib.GetFaceChipDetails(shape, 150, 0.25);
                    var faceChip = Dlib.ExtractImageChip<RgbPixel>(img, faceChipDetail);

                    // convert the chip to a matrix and store
                    var matrix = new Matrix<RgbPixel>(faceChip);
                    chips.Add(matrix);
                    faces.Add(face);
                }
                Console.WriteLine($"   Found {chips.Count} faces in image");
                // put each fae in a 128D embedding space
                // similar faces will be placed close together
                Console.WriteLine("Recognizing faces...");
                var descriptors = dnn.Operator(chips);

                // compare each face with all other faces
                var edges = new List<SamplePair>();
                for (uint i = 0; i < descriptors.Count; ++i)
                    for (var j = i; j < descriptors.Count; ++j)

                        // record every pair of two similar faces
                        // faces are similar if they are less than 0.6 apart in the 128D embedding space
                        if (Dlib.Length(descriptors[i] - descriptors[j]) < 0.6)
                            edges.Add(new SamplePair(i, j));
                // use the chinese whispers algorithm to find all face clusters
                Dlib.ChineseWhispers(edges, 100, out var clusters, out var labels);
                Console.WriteLine($"   Found {clusters} unique person(s) in the image");

                // create a color palette for plotting
                // Tentukan jumlah warna yang diinginkan
                int numColors = 50;

                // Buat sebuah array untuk menampung warna acak
                var palette = new RgbPixel[numColors];

                // Buat objek Random
                var random = new Random();

                // Isi array dengan warna acak
                for (int i = 0; i < numColors; i++)
                {
                    // Generate komponen warna secara acak antara 0 dan 255
                    byte r = (byte)random.Next(256);
                    byte g = (byte)random.Next(256);
                    byte b = (byte)random.Next(256);

                    // Tambahkan warna acak ke dalam array
                    palette[i] = new RgbPixel(r, g, b);
                }

                // draw rectangles on each face using the cluster color
                for (var i = 0; i < faces.Count; i++)
                {
                    Dlib.DrawRectangle(img, faces[i], color: palette[labels[i]], thickness: 4);
                }

                // create the output folder if it doesn't exist
                var outputFolder = "output";
                Directory.CreateDirectory(outputFolder);
                // export the modified image
                Dlib.SaveJpeg(img, "output/output.jpg");
            }

        }
    }
}
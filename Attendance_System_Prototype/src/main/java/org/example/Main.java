package org.example;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        nu.pattern.OpenCV.loadLocally();

        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println("Error: Camera not found.");
            return;
        }

        CascadeClassifier faceDetector = new CascadeClassifier("C:\\Users\\hp\\Downloads\\Attendance_System_Prototype\\src\\main\\java\\org\\example\\haarcascade_frontalface_default.xml");
        if (faceDetector.empty()) {
            System.out.println("Error: Cascade classifier not found.");
            return;
        }

        ComputationGraph model = loadKerasModel("C:\\Users\\hp\\Downloads\\functional_model.h5");

        // Labels mapping
        Map<Integer, String> labels = new HashMap<>();
        labels.put(0, "Ahhan");
        labels.put(1, "Aryan");
        labels.put(2, "Pratyaksh");
        labels.put(3, "Suryansh");

        while (true) {
            Mat frame = new Mat();
            camera.read(frame);

            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(frame, faceDetections, 1.17, 5, 0, new Size(30, 30), new Size());

            for (Rect rect : faceDetections.toArray()) {
                Mat face = new Mat(frame, rect);

                // Convert OpenCV Mat to BufferedImage
                BufferedImage bufferedImage = matToBufferedImage(face);

                // Resize and preprocess the image for model input
                INDArray input = imageToINDArray(bufferedImage);

                // Perform prediction
                INDArray[] outputs = model.output(input);

                // Get the top predictions
                int[] topPredictions = Nd4j.argMax(outputs[0], 1).toIntVector();
                double[] predictionProbabilities = outputs[0].toDoubleVector();

                // Display the predicted labels
                for (int i = 0; i < topPredictions.length; i++) {
                    int predictedLabel = topPredictions[i];
                    String predictedName = labels.get(predictedLabel);
                    double probability = predictionProbabilities[predictedLabel];

                    // Round the probability to two decimals
                    String formattedProbability = String.format("%.2f", probability);

                    // Draw rectangle around the detected face
                    Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(0, 255, 0), 2);

                    // Display the predicted label with rounded probability
                    String label = "Prediction: " + predictedName + " (Probability: " + formattedProbability + ")";
                    Imgproc.putText(frame, label, new Point(rect.x, rect.y - 10 - i * 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
                            new Scalar(0, 255, 0), 2);
                }

                // Check for ambiguity
                if (topPredictions.length > 1) {
                    System.out.println("Ambiguous Prediction");
                }
            }

            HighGui.imshow("Face Detection", frame);
            HighGui.waitKey(1);


            try {
                if (System.in.available() > 0 && System.in.read() == 'q') {
                    break;
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }

    private static ComputationGraph loadKerasModel(String modelPath) {
        try {
            try {
                return KerasModelImport.importKerasModelAndWeights(modelPath);
            } catch (UnsupportedKerasConfigurationException e) {
                throw new RuntimeException(e);
            } catch (InvalidKerasConfigurationException e) {
                throw new RuntimeException(e);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error loading Keras model", e);
        }
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage bufferedImage = new BufferedImage(mat.width(), mat.height(), type);
        mat.get(0, 0, ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData());
        return bufferedImage;
    }

    private static INDArray imageToINDArray(BufferedImage image) {
        // Resize the image to the desired dimensions (64x64)
        BufferedImage resizedImage = resizeImage(image, 128, 128);

        // Convert the BufferedImage to a 3D INDArray (height x width x channels)
        int channels = 3; // Assuming RGB image
        int height = resizedImage.getHeight();
        int width = resizedImage.getWidth();

        INDArray indArray = Nd4j.create(1, height, width, channels);

        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    Color color = new Color(resizedImage.getRGB(w, h));
                    indArray.putScalar(0, h, w, c, color.getRed() / 255.0); // Normalize pixel values to [0, 1]
                }
            }
        }

        return indArray;
    }

    private static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        Image resultingImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_DEFAULT);
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(resultingImage, 0, 0, null);
        g2d.dispose();
        return resizedImage;
    }
}

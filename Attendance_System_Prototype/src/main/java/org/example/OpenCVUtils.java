package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import org.nd4j.linalg.api.buffer.DataType;

public class OpenCVUtils {
    public static INDArray matToINDArray(Mat mat) {
        // Convert OpenCV Mat to ND4J INDArray

        // Convert BGR to RGB
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);

        // Convert Mat to float
        MatOfFloat matOfFloat = new MatOfFloat(1.0f);
        MatOfInt histSize = new MatOfInt(256);
        MatOfFloat ranges = new MatOfFloat(0, 256);
        Imgproc.calcHist(Arrays.asList(mat), new MatOfInt(0), new Mat(), matOfFloat, histSize, ranges);
        mat.convertTo(mat, CvType.CV_32F, 1.0 / mat.total());

        // Normalize values to range [0, 1]
        mat = mat.reshape(1, 1);

        // Reshape Mat to 3D array
        int[] shape = new int[]{1, mat.channels(), mat.rows(), mat.cols()};
        INDArray imageArray = Nd4j.create(mat.get(0, 0));

        // Add a batch dimension
        return imageArray;
    }

    public static Mat drawText(Mat image, String text, Point position, Scalar color) {
        Imgproc.putText(image, text, position, Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
        return image;
    }

    public static Mat drawRectangle(Mat image, Rect rect, Scalar color) {
        Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), color, 2);
        return image;
    }
}

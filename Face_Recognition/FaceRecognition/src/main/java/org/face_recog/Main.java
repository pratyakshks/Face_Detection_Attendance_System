package org.face_recog;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Core;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.Random;
import org.deeplearning4j.util.ModelSerializer;

public class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws Exception {
        nu.pattern.OpenCV.loadLocally();

        // Set your image directory path
        File baseDir = new File("C:\\Users\\hp\\Downloads\\faces");
        FileSplit fileSplit = new FileSplit(baseDir, NativeImageLoader.ALLOWED_FORMATS, new Random());

        int numClasses = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        System.out.println(numClasses);

        // Define image parameters
        int height = 224;
        int width = 224;
        int channels = 3;

        // Define iteration configurations
        int batchSize = 16;

        // Image transformation for resizing during training
        MultiImageTransform multiImageTransform = new MultiImageTransform(new ResizeImageTransform(height, width));

        // Initialize the ImageRecordReader with resizing
        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
        imageRecordReader.initialize(fileSplit, multiImageTransform);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, numClasses);
        trainIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        // Define the neural network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(1e-4))
                .list()
                .layer(new ConvolutionLayer.Builder().nIn(channels).nOut(64)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU).build())
                .layer(new ConvolutionLayer.Builder().nOut(128)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(100)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses).weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        // Initialize the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println("Model Built Successfully");
        // Attach a listener for scoring (accuracy) after every iteration
        model.setListeners(new ScoreIterationListener(1));

        // Train the model
        int numEpochs = 10;
        for (int i = 0; i < numEpochs; i++) {
            System.out.println(i);
            trainIter.reset(); // Reset iterator for each epoch
            model.fit(trainIter);
        }

        // Evaluate the model on the training data
        Evaluation eval = new Evaluation(numClasses);
        trainIter.reset();
        while (trainIter.hasNext()) {
            org.nd4j.linalg.dataset.DataSet ds = trainIter.next();
            INDArray output = model.output(ds.getFeatures(), false);
            eval.eval(ds.getLabels(), output);
        }

        // Print the evaluation metrics
        log.info(eval.stats());
        File locationToSave = new File("model.zip");
        ModelSerializer.writeModel(model, locationToSave, true);
        log.info("Model saved successfully at: {}", locationToSave.getAbsolutePath());
    }
}

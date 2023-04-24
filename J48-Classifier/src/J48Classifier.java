import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
// Java Program for Creating a Model Based on J48 Classifier
// Importing required classes
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class J48Classifier {
    public static double getHammingLoss(Evaluation evaluation, int numLabels) {
        double hammingLoss = 0.0;
        double[][] confusionMatrix = evaluation.confusionMatrix();
        for (int i = 0; i < confusionMatrix.length; i++) {
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                if (i != j) {
                    hammingLoss += confusionMatrix[i][j] / numLabels;
                }
            }
        }
        return hammingLoss / evaluation.numInstances();
    }

    public static double getExactMatchRatio(Evaluation evaluation) {
        double exactMatchRatio = 0.0;
        double[][] confusionMatrix = evaluation.confusionMatrix();
        for (int i = 0; i < confusionMatrix.length; i++) {
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                if (i == j) {
                    exactMatchRatio += confusionMatrix[i][j];
                }
            }
        }
        return exactMatchRatio / evaluation.numInstances();
    }

    public static void evaluateFile(String dataset){
        try{
            BufferedReader bufferedReader = new BufferedReader(new FileReader(dataset));
            // Creating J48 classifier
            J48 j48Classifier = new J48();
            j48Classifier.setUnpruned(true);
            // Create dataset instances
            Instances datasetInstances = new Instances(bufferedReader);

            // Set Target Class
            datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);

            // Evaluation
            Evaluation evaluation = new Evaluation(datasetInstances);

            // Bagging
            Bagging bagger = new Bagging();
            bagger.setClassifier(j48Classifier);

            // Boosting
            AdaBoostM1 adaBoost = new AdaBoostM1();
            adaBoost.setClassifier(j48Classifier);

            // Unpruned Tree
            evaluation.crossValidateModel(j48Classifier, datasetInstances, 10, new Random(1));
            System.out.println(evaluation.toSummaryString(
                    "\nResults: Unpruned", false));
            System.out.println("Hamming Loss: " + getHammingLoss(evaluation, datasetInstances.numClasses()));
            System.out.println("Exact Match Ratio: " + getExactMatchRatio(evaluation));
            System.out.println(evaluation.toClassDetailsString());

            // Pruned Tree
            J48 j48PrunedClassifier = new J48();
            j48PrunedClassifier.setUnpruned(false);
            evaluation.crossValidateModel(j48PrunedClassifier, datasetInstances, 10, new Random(1));
            System.out.println(evaluation.toSummaryString(
                    "\nResults: Pruned", false));
            System.out.println("Hamming Loss: " + getHammingLoss(evaluation, datasetInstances.numClasses()));
            System.out.println("Exact Match Ratio: " + getExactMatchRatio(evaluation));
            System.out.println(evaluation.toClassDetailsString());

            // Bagging-unpruned
            evaluation.crossValidateModel(bagger, datasetInstances, 10, new Random(1));
            System.out.println(evaluation.toSummaryString(
                    "\nResults: Bagging-UnPruned", false));
            System.out.println("Hamming Loss: " + getHammingLoss(evaluation, datasetInstances.numClasses()));
            System.out.println("Exact Match Ratio: " + getExactMatchRatio(evaluation));
            System.out.println(evaluation.toClassDetailsString());

            // Boosting-unpruned
            evaluation.crossValidateModel(adaBoost, datasetInstances, 10, new Random(1));
            System.out.println(evaluation.toSummaryString(
                    "\nResults: Boosting-Unpruned", false));
            System.out.println("Hamming Loss: " + getHammingLoss(evaluation, datasetInstances.numClasses()));
            System.out.println("Exact Match Ratio: " + getExactMatchRatio(evaluation));
            System.out.println(evaluation.toClassDetailsString());

            // Bagging-pruned
            Bagging baggerPruned = new Bagging();
            baggerPruned.setClassifier(j48PrunedClassifier);
            evaluation.crossValidateModel(baggerPruned, datasetInstances, 10, new Random(1));
            System.out.println(evaluation.toSummaryString(
                    "\nResults: Bagging-Pruned", false));
            System.out.println("Hamming Loss: " + getHammingLoss(evaluation, datasetInstances.numClasses()));
            System.out.println("Exact Match Ratio: " + getExactMatchRatio(evaluation));
            System.out.println(evaluation.toClassDetailsString());

            // Boosting-pruned
            AdaBoostM1 adaBoostPruned = new AdaBoostM1();
            adaBoostPruned.setClassifier(j48PrunedClassifier);
            evaluation.crossValidateModel(adaBoostPruned, datasetInstances, 10, new Random(1));
            System.out.println(evaluation.toSummaryString(
                    "\nResults: Boosting-Pruned", false));
            System.out.println("Hamming Loss: " + getHammingLoss(evaluation, datasetInstances.numClasses()));
            System.out.println("Exact Match Ratio: " + getExactMatchRatio(evaluation));
            System.out.println(evaluation.toClassDetailsString());
        }catch (Exception e){
            System.out.println("Error Occurred!!!! \n"
                    + e.getMessage());
        }
    }

    // Main driver method
    public static void main(String args[]) {
        // Try block to check for exceptions
        try {
            evaluateFile("./data/exported/method_mld_fe_lc.arff");
        }

        // Catch block to check for rexceptions
        catch (Exception e) {

            // Print and display the display message
            // using getMessage() method
            System.out.println("Error Occurred!!!! \n"
                    + e.getMessage());
        }

        // Display message to be printed ion console
        // when program is successfully executed
    }
}

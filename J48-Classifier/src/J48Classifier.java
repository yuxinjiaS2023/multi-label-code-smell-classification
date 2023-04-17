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
    // Main driver method
    public static void main(String args[]) {
        // Try block to check for exceptions
        try {
            // Creating J48 classifier
            J48 j48Classifier = new J48();

            // Dataset path
            String dataset = "./data/exported/method_mld_no_fe_lc.arff";

            // Create bufferedreader to read the dataset
            BufferedReader bufferedReader = new BufferedReader(new FileReader(dataset));

            // Create dataset instances
            Instances datasetInstances = new Instances(bufferedReader);

            // Set Target Class
            datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);

            // Evaluation
            Evaluation evaluation = new Evaluation(datasetInstances);

            // Bagging
            Bagging bagger = new Bagging();
            bagger.setClassifier(j48Classifier);
            bagger.setSeed(1);

            // Boosting
            AdaBoostM1 adaBoost = new AdaBoostM1();
            adaBoost.setClassifier(j48Classifier);

            // Cross Validate Model with 10 folds
            evaluation.crossValidateModel(bagger, datasetInstances, 10, new Random(1));
            System.out.println(evaluation.toSummaryString(
                    "\nResults", false));
            System.out.println(evaluation.toClassDetailsString());
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

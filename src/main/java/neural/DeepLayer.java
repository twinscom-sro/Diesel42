package neural;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A simple implementation of a fully connected Deep Neural Network
 * designed to demonstrate correct backpropagation, including bias updates.
 * This network uses the Sigmoid activation function and trains on the XOR problem.
 *
 * Architecture: 2 (Input) -> 3 (Hidden) -> 1 (Output)
 */
public class DeepLayer {

    // --- Hyperparameters ---
    public static double LEARNING_RATE = 0.5;
    private static final Random RANDOM = new Random(42); // Seed for reproducible results

    // --- Network Structure (Layer sizes) ---
    private final int[] layerSizes;

    // --- Network Parameters (Weights and Biases) ---
    // weights[l][j][i]: weight from neuron i in layer l to neuron j in layer l+1
    private double[][][] weights;
    // biases[l][j]: bias for neuron j in layer l+1
    private double[][] biases;

    // --- Network State (During Forward/Backward Pass) ---
    // activations[l][j]: output activation of neuron j in layer l
    private double[][] activations;
    // z_values[l][j]: weighted sum (input) to neuron j in layer l (before activation)
    private double[][] z_values;
    // deltas[l][j]: error term (delta) for neuron j in layer l
    private double[][] deltas;

    /**
     * Constructs the neural network with specified layer sizes.
     * @param layerSizes An array of layer sizes (e.g., {2, 3, 1} for XOR).
     */
    public DeepLayer(int... layerSizes) {
        this.layerSizes = layerSizes;
        initializeParameters();
    }

    // --- Initialization ---

    private void initializeParameters() {
        // L layers means L-1 sets of weights/biases (connections between layers)
        int numWeightLayers = layerSizes.length - 1;

        weights = new double[numWeightLayers][][];
        biases = new double[numWeightLayers][];

        // Initialize weights and biases
        for (int l = 0; l < numWeightLayers; l++) {
            int prevLayerSize = layerSizes[l];
            int currentLayerSize = layerSizes[l + 1];

            // Initialize biases for the current layer (l+1)
            biases[l] = new double[currentLayerSize];
            // Initialize weights connecting layer l to layer l+1
            weights[l] = new double[currentLayerSize][prevLayerSize];

            // Use Xavier/Glorot-like initialization for better training stability
            double limit = Math.sqrt(2.0 / (prevLayerSize + currentLayerSize));

            for (int j = 0; j < currentLayerSize; j++) {
                // Initialize bias for neuron j in layer l+1
                biases[l][j] = RANDOM.nextDouble() * 2 * limit - limit;
                for (int i = 0; i < prevLayerSize; i++) {
                    // Initialize weight connecting neuron i (prev) to neuron j (current)
                    weights[l][j][i] = RANDOM.nextDouble() * 2 * limit - limit;
                }
            }
        }
    }

    // --- Activation Functions ---

    /** The Sigmoid activation function. */
    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /** The derivative of the Sigmoid function. */
    private static double sigmoidDerivative(double a) {
        // a is the output (activation) of the neuron
        return a * (1.0 - a);
    }

    // --- Forward Propagation ---

    /**
     * Executes the forward pass through the network.
     * @param inputVector The initial input array.
     * @return The output of the network (last layer's activations).
     */
    public double[] feedForward(double[] inputVector) {
        // Initialize state arrays for activations and z_values
        activations = new double[layerSizes.length][];
        z_values = new double[layerSizes.length][];

        // Set the input layer activations (Layer 0)
        activations[0] = inputVector;

        // Iterate through each weight layer (l=0 is connection from Layer 0 to Layer 1)
        for (int l = 0; l < weights.length; l++) {
            double[] prevActivation = activations[l];
            int currentLayerSize = layerSizes[l + 1];

            z_values[l + 1] = new double[currentLayerSize];
            activations[l + 1] = new double[currentLayerSize];

            // Iterate over neurons in the current layer (j)
            for (int j = 0; j < currentLayerSize; j++) {
                double z = biases[l][j]; // Start with the bias

                // Sum weighted inputs from the previous layer (i)
                for (int i = 0; i < prevActivation.length; i++) {
                   // System.out.format("weights[%d][%d][%d] * prevActivation[%d]\n",l,j,i,i);
                    z += weights[l][j][i] * prevActivation[i];
                }

                z_values[l + 1][j] = z;
                activations[l + 1][j] = sigmoid(z);
            }
        }

        // Return the activations of the final layer
        return activations[layerSizes.length - 1];
    }

    // --- Backpropagation and Training ---

    /**
     * Trains the network on a single input/target pair using backpropagation.
     * @param input The training input vector.
     * @param target The expected output vector.
     */
    public void train(double[] input, double target) {
        // 1. Forward Pass to calculate activations and z_values
        feedForward(input);

        // 2. Backward Pass (Error calculation)
        double[] _target = new double[1];
        _target[0] = target;
        backpropagate(_target);

        // 3. Update Weights and Biases
        updateParameters();
    }

    /**
     * Calculates the error terms (deltas) for all layers using backpropagation.
     * @param target The expected output vector.
     */
    public void backpropagate(double[] target) {
        int L = layerSizes.length - 1; // Index of the last layer (output layer)
        deltas = new double[layerSizes.length][];

        // --- Step 1: Calculate Error Delta for the Output Layer (L) ---
        deltas[L] = new double[layerSizes[L]];

        for (int j = 0; j < layerSizes[L]; j++) {
            double activation = activations[L][j];
            // Error term: (Output - Target) * sigmoid'(z)
            // Note: MSE derivative is (a - y)
            deltas[L][j] = (activation - target[j]) * sigmoidDerivative(activation);
        }

        // --- Step 2: Backpropagate Error Deltas through Hidden Layers ---
        // Iterate backwards from the second-to-last layer (L-1) down to the first hidden layer (1)
        for (int l = L - 1; l >= 1; l--) {
            int currentLayerSize = layerSizes[l];
            int nextLayerSize = layerSizes[l + 1];
            deltas[l] = new double[currentLayerSize];

            // Weights connecting layer l to layer l+1 are stored at weights[l]
            double[][] w_l_to_lplus1 = weights[l];
            double[] delta_lplus1 = deltas[l + 1];

            // Iterate over neurons in the current layer (j)
            for (int j = 0; j < currentLayerSize; j++) {
                // Sum of weighted errors from the next layer
                double weightedErrorSum = 0;
                for (int k = 0; k < nextLayerSize; k++) {
                    // w_l_to_lplus1[k][j]: weight from neuron j (current layer l) to neuron k (next layer l+1)
                    weightedErrorSum += w_l_to_lplus1[k][j] * delta_lplus1[k];
                }

                // Error term: (Sum of weighted errors) * sigmoid'(z)
                double activation = activations[l][j];
                deltas[l][j] = weightedErrorSum * sigmoidDerivative(activation);
            }
        }
    }

    /**
     * Applies the weight and bias updates across the entire network.
     */
    public void updateParameters() {
        // Iterate through each set of weights/biases (l=0 is the first connection)
        for (int l = 0; l < weights.length; l++) {
            double[] prevActivation = activations[l]; // Activations a^(l)
            double[] delta = deltas[l + 1];          // Error terms delta^(l+1)

            // Update Biases (l+1 layer)
            for (int j = 0; j < layerSizes[l + 1]; j++) {
                // Change in bias: -eta * delta^(l+1)_j
                biases[l][j] -= LEARNING_RATE * delta[j];
            }

            // Update Weights (l to l+1 connection)
            for (int j = 0; j < layerSizes[l + 1]; j++) { // Current neuron index (l+1)
                for (int i = 0; i < layerSizes[l]; i++) {  // Previous neuron index (l)
                    // Change in weight: -eta * delta^(l+1)_j * a^l_i
                    weights[l][j][i] -= LEARNING_RATE * delta[j] * prevActivation[i];
                }
            }
        }
    }

    /**
     * Calculates the total Mean Squared Error (MSE) for a given output and target.
     * @param output The network's actual output.
     * @param target The expected target output.
     * @return The total MSE.
     */
    public double calculateMSE(double[] output, double[] target) {
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            sum += Math.pow(target[i] - output[i], 2);
        }
        return sum / output.length;
    }

    public static List<String[]> readCsv(String filePath) throws IOException {
        List<String[]> records = new ArrayList<>();
        String line;
        String DEFAULT_DELIMITER = ",";

        // Use try-with-resources to ensure the BufferedReader is closed automatically
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            while ((line = br.readLine()) != null) {
                // Simple parsing using String.split() is often sufficient,
                // but a more robust method is needed for quoted fields.
                // For simplicity here, we use a basic regex split that handles quoting better
                // than simple String.split(","):
                // Lookbehind for a comma not preceded by an even number of quotes,
                // followed by a comma, not followed by an even number of quotes.
                // NOTE: For enterprise applications, use Apache Commons CSV or similar dedicated library.
                String[] values = line.split(DEFAULT_DELIMITER + "(?=([^\"]*\"[^\"]*\")*[^\"]*$)");

                String year = values[0].substring(0,4);
                if( year.contentEquals("2021") || year.contentEquals("2022") || year.contentEquals("2023") | year.contentEquals("2024") ) {
                    // Clean up any residual quotes from the split process
                    int pos = 0;
                    String[] v = new String[values.length - 2];
                    v[pos++] = values[1];
                    for (int i = 3; i < values.length; i++) {
                        v[pos++] = values[i].trim().replace("\"", "");
                    }
                    records.add(v);
                }
            }
        }
        return records;
    }

    void trainingResults(DeepLayer network, double[][] X, double[][] Y){
        int TT=0;
        int FF=0;
        int TF=0;
        int FT=0;

        for (int i = 0; i < X.length; i++) {
            double[] output = network.feedForward(X[i]);
            double mse = network.calculateMSE(output, Y[i]);
            //System.out.format("Input: %s | Target: %.2f | Predicted: %.4f | MSE: %.6f\n",
            //        Arrays.toString(X[i]), Y[i][0], output[0], mse);
            boolean exp = Y[i][0]>0.5;
            boolean ach = output[0]>0.5;

            if( exp && ach ) TT++;
            if( !exp && !ach ) FF++;
            if( !exp && ach ) FT++;
            if( exp && !ach ) TF++;
        }

        System.out.format("TT=%d, TF=%d, FT=%d, FF=%d\n", TT, TF, FT, FF);

    }

 }


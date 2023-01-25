package ch.sthomas.neuralnetwork;

import java.io.*;
import java.util.Arrays;

public class Network implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    public static final ActivationType ACTIVATION_TYPE = ActivationType.SIGMOID;

    private final Neuron[][] neurons;

    private final int nrOfInputs;

    /**
     * Create a new Network.
     *
     * @param nrOfInputs the number of inputs for the network
     * @param inputs     the number of input Neurons for the network
     * @param hidden     the number of hidden Neurons for the network
     * @param outputs    the number of output Neurons for the network
     */
    public Network(int nrOfInputs, int inputs, int[] hidden, int outputs) {
        this.nrOfInputs = nrOfInputs;
        this.neurons = new Neuron[hidden.length + 2][];

        // Create input  neuron layers
        this.neurons[0] = new Neuron[inputs];
        // Create hidden neuron layers
        for (int i = 0; i < hidden.length; i++) {
            this.neurons[i + 1] = new Neuron[hidden[i]];
        }
        // Create output neuron layers after input layer and hidden layers
        this.neurons[hidden.length + 1] = new Neuron[outputs];

        this.initInputs();
    }

    public void initInputs() {
        for (int i = 0; i < this.neurons.length; i++) {
            for (int j = 0; j < this.neurons[i].length; j++) {
                int nrOfWeights = i == 0 ? this.nrOfInputs : this.neurons[i - 1].length;
                this.neurons[i][j] = new Neuron(MathUtils.getRandomDoubles(nrOfWeights), 0);
            }
        }
    }

    /**
     * Calculate the output values for the given inputs.
     *
     * @param inputs the inputs to the first layer
     * @return an array of length neurons[neurons.length - 1].length containing all the values the output layer produced
     */
    public double[] calculateOutputs(double[] inputs) {
        for (Neuron[] layer : this.neurons) {
            // Initialize array for the newly calculated inputs for the next layer
            double[] newInputs = new double[layer.length];
            // Calculate the new values
            for (int i = 0; i < layer.length; i++) {
                newInputs[i] = layer[i].activate(inputs);
            }
            // Override the old inputs with the new ones
            inputs = newInputs;
        }

        return inputs;
    }

    /**
     * Calculate the output values for the given inputs.
     *
     * @param inputs the inputs to the first layer
     * @return an array of length neurons[neurons.length - 1].length containing all the values the output layer produced
     */
    public double[][] calculateAllNeuronActivations(double[] inputs) {
        double[][] result = new double[this.neurons.length][];

        for (int i = 0; i < this.neurons.length; i++) {
            Neuron[] layer = this.neurons[i];
            // Initialize array for the newly calculated inputs for the next layer
            double[] newInputs = new double[layer.length];
            // Calculate the new values
            for (int j = 0; j < layer.length; j++) {
                newInputs[j] = layer[j].activate(inputs);
            }
            // Override the old inputs with the new ones
            inputs = newInputs;

            result[i] = inputs;
        }

        return result;
    }

    public void train(double[][] inputs, double[][] expectedOutput) {
        this.backPropagationTrain(inputs, expectedOutput);
    }

    /**
     * Train the neural network so the result(inputs) tends toward the {@code expectedOutput}.
     *
     * @param inputs         the inputs to train on, first dimension = time, then the input vector of values
     * @param expectedOutput the expected outputs to the given inputs, same dimensions as the {@code inputs}
     */
    public void backPropagationTrain(double[][] inputs, double[][] expectedOutput) {
        if (inputs.length != expectedOutput.length)
            throw new IllegalArgumentException("Back Propagation can only be calculated for vectors of the same length.");

        double[][][] activation = new double[expectedOutput.length][inputs.length][];

        for (int i = 0; i < inputs.length; i++) {
            activation[i] = calculateAllNeuronActivations(inputs[i]);
        }

        // Calculate cost gradients and add them to the weights
        for (int i = 0; i < this.neurons.length; i++) {
            for (int j = 0; j < this.neurons[i].length; j++) {
                double[] costGradient = this.getCostGradient(inputs, expectedOutput, activation, i, j);
                this.neurons[i][j].addToWeights(costGradient);
            }
        }

        // TODO: Backpropagation missing
    }

    /**
     * Save a serialized network to the disk.
     *
     * @param filename the filename, the extension '.network' will be appended
     * @throws IOException when the file opening throws this exception
     */
    public void saveToDisk(String filename) throws IOException {
        String filepath = System.getProperty("user.dir") + filename + ".network";
        String dirpath = filepath.substring(0, Math.max(Math.max(filepath.lastIndexOf("/"), filepath.lastIndexOf("\"")), filepath.lastIndexOf(File.separator)));
        File dir = new File(dirpath);
        File file = new File(filepath);

        if (!file.exists()) {
            dir.mkdirs();
            file.createNewFile();
        }
        try (FileOutputStream outStream = new FileOutputStream(filepath); ObjectOutputStream oOutStream = new ObjectOutputStream(outStream)) {
            oOutStream.writeObject(this);
            oOutStream.flush();
        }
    }

    /**
     * Read a serialized network from the disk.
     *
     * @param filename the filename, the extension '.network' will be appended
     * @return the network from disk
     * @throws IOException            when the file opening throws this exception
     * @throws ClassNotFoundException when the file does not match a known class
     */
    public static Network readFromDisk(String filename) throws IOException, ClassNotFoundException {
        String filepath = System.getProperty("user.dir") + filename + ".network";
        try (FileInputStream inStream = new FileInputStream(filepath); ObjectInputStream oInStream = new ObjectInputStream(inStream)) {
            return (Network) oInStream.readObject();
        }
    }

    /**
     * @param actual   the actual results
     * @param expected the expected values
     * @return the sum of the cost vector's fields
     */
    public double getCostValue(double[] actual, double[] expected) {
        return Arrays.stream(this.getCostVector(actual, expected)).sum();
    }

    /**
     * Returns a vector of the squared difference of the respective expected and actual value.
     * {@code actual} must have the same size as {@code expected}.
     *
     * @param actual   the actual values
     * @param expected the expected values
     * @return the cost vector
     */
    private double[] getCostVector(double[] actual, double[] expected) {
        if (actual.length != expected.length)
            throw new IllegalArgumentException("Cost can only be calculated for vectors of the same length.");
        double[] result = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            result[i] = this.getCost(actual[i], expected[i]);
        }
        return result;
    }

    private double getCost(double actual, double expected) {
        return Math.pow(actual - expected, 2);
    }

    /**
     * @param input       the training input data that leads to the activations array
     * @param activations the 3d-matrix with [Time][Layer][Neuron]
     * @param layer       the second index for activations
     * @param node        the third index for activations
     * @return the cost gradient for the specific node at the specific time
     */
    private double[] getCostGradient(double[][] input, double[][] expected, double[][][] activations, int layer, int node) {
        // input[0].length == input[1].length == input[n].length, the index only changes the time so input[0].length == nr of inputs per data set
        int previousLayerNeuronsLength = layer > 0 ? this.neurons[layer - 1].length : input[0].length;

        double[][] aL1k = new double[previousLayerNeuronsLength][activations.length]; // [i][j]; i = incoming edge with activation value of the previous neuron, j = time
        double[] ds = new double[activations.length];
        double[] magnitude = new double[activations.length];
        // Calculate d C / d w^L_jk
        for (int time = 0; time < activations.length; time++) {
            // Handle layer == 0
            double[] previousLayerActivations = layer > 0 ? activations[time][layer - 1] : input[time];

            // Calculate a^(L-1)_k = dz_L / dw_L
            for (int k = 0; k < aL1k.length; k++) aL1k[k][time] = previousLayerActivations[k];
            // Calculate sigmoid'(z^L_j) = da_L / dz_L
            ds[time] = Neuron.sigmoidDerivative(this.neurons[layer][node], input[time]);
            // Calculate 2 * (a^L - expected) = dC_0/da_L
            magnitude[time] = -2 * (activations[time][layer][node] - expected[time][node]);

//            System.out.println(activations[time][layer][node] + ", " + expected[time][node] + " so resulting in " + aL1k[0][time] + "*" + ds[time] + "*" + magnitude[time] + " = " + aL1k[0][time] * ds[time] * magnitude[time]);
        }

        // Recurse the previous layer with the newly calculated expected values
        double[] res = new double[aL1k.length];
        for (int i = 0; i < res.length; i++) {
            for (int j = 0; j < activations.length; j++) {
                res[i] += (aL1k[i][j] * ds[j] * magnitude[j]);
            }

            // 1. Multiply the Training effectiveness and 2. Take the average instead of the sum
//            res[i] *= (Neuron.TRAINING_ALPHA / activations.length);
            res[i] /= activations.length;
        }
        return res;
    }
}

enum ActivationType {
    LINEAR, // Unused, Cost Gradient must be updated to support Linear ActivationType
    SIGMOID // Requires a more complex learning algorithm with derivatives, see https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e
}

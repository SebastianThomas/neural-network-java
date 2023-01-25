package ch.sthomas.neuralnetwork;

import java.util.Arrays;

public class Network {
    public static final ActivationType ACTIVATION_TYPE = ActivationType.LINEAR;

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
                this.neurons[i][j] = new Neuron(MathUtils.getRandomDoubles(nrOfWeights), 0, nrOfWeights / 2.5);
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
            System.out.println(Arrays.toString(layer));
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

    public void train(double[] inputs, double[] expectedOutput) {
        this.backPropagationTrain(inputs, expectedOutput);
    }

    public void backPropagationTrain(double[] inputs, double[] expectedOutput) {
        // TODO: Implement back propagation
    }
}

enum ActivationType {
    LINEAR, SIGMOID // Requires a more complex learning algorithm with derivatives, see https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e
}

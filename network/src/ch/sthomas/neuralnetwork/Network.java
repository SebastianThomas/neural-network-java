package ch.sthomas.neuralnetwork;

public class Network {
    public static final ActivationType ACTIVATION_TYPE = ActivationType.SIGMOID;

    private final Neuron[][] neurons;

    public Network(int inputs, int[] hidden, int outputs) {
        this.neurons = new Neuron[hidden.length + 2][];

        // Create input  neuron layers
        this.neurons[0] = new Neuron[inputs];
        // Create hidden neuron layers
        for (int i = 0; i < hidden.length; i++) {
            this.neurons[i + 1] = new Neuron[hidden[i]];
        }
        // Create output neuron layers after input layer and hidden layers
        this.neurons[hidden.length + 1] = new Neuron[outputs];
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
}

enum ActivationType {
    LINEAR,
    SIGMOID
}

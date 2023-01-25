package ch.sthomas.neuralnetwork;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

public class Neuron implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * The value of alpha for training (to avoid overshooting when correcting a value.
     */
    public static final double TRAINING_ALPHA = 0.01;
    /**
     * The maximum deviation that expected and actual result may have during training the Neuron.
     */
    public static final double MAX_TRAINING_DEVIATION = 0.1;

    private double[] weights;
    private double bias;
    /**
     * Unused
     */
    private double activationThreshold;

    /**
     * Create a new Neuron with
     *
     * @param weights the initial weights of the Neuron
     * @param bias    the initial bias of the Neuron
     * @see Neuron#addToWeights(double[])
     * @see Neuron#addToBias(double)
     */
    public Neuron(double[] weights, double bias) {
        this.weights = weights;
        this.bias = bias;
        this.activationThreshold = this.calcActivationThreshold();
    }

    /**
     * Train the Neuron with a perception-algorithm.
     *
     * @param inputs         the input that should resolve to the expected result
     * @param expectedResult the expectation of the result the Neuron should be trained towards
     * @see Neuron#TRAINING_ALPHA
     * @see Neuron#MAX_TRAINING_DEVIATION
     */
    public void train(double[] inputs, double expectedResult) {
        double actualResult = activate(inputs);
        while (Math.abs(actualResult - expectedResult) > MAX_TRAINING_DEVIATION) {
            for (int i = 0; i < this.weights.length; i++) {
                this.weights[i] += TRAINING_ALPHA * inputs[i] * (expectedResult - actualResult);
            }
            actualResult = activate(inputs);
        }
    }

    /**
     * Fire the neuron if it exceeds its activationValue given the input values. Uses the algorithm specified in
     * {@link Network#ACTIVATION_TYPE}.
     *
     * @param inputs from the previous layer or program inputs
     * @return 1.0 if it fires and 0.0 if it does not
     * @see Network#ACTIVATION_TYPE
     */
    public double activate(double[] inputs) {
        return calculateValue(inputs); // > this.activationValue ? 1.0 : 0.0;
    }

    /**
     * Calculate the value of the neuron. Represents {@code a}, the activation value, in the 3B1B-video.
     *
     * @param inputs the inputs to calculate the value for
     * @return the value of the neuron given the inputs
     * @see Network#ACTIVATION_TYPE
     */
    private double calculateValue(double[] inputs) {
        // If inputs have more value than the weights or the other way around, the resulting values should just be 0, so they can just be ignored
        return switch (Network.ACTIVATION_TYPE) {
            case LINEAR -> linearActivation(inputs);
            case SIGMOID -> sigmoidActivation(inputs);
        };
    }

    /**
     * Set the weights to a new array. May be of different size than the current weights-array.
     *
     * @param weights the new weights
     * @see Neuron#addToWeights(double[])
     */
    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    /**
     * Add the respective values to the current weights
     *
     * @param values the values to be added to the weights, values[x] = 0 for no change to edge x
     * @see Neuron#setWeights(double[])
     */
    public void addToWeights(double[] values) {
        if (this.weights.length != values.length)
            throw new IllegalArgumentException("Length of weights to change must match the already existing weights' length.");
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] += values[i];
        }
    }

    /**
     * Set the bias to a new value.
     *
     * @param bias the new value for the bias
     * @see Neuron#addToBias(double)
     */
    public void setBias(double bias) {
        this.bias = bias;
    }

    /**
     * Add a value to the bias. To reduce the bias, value may be negative.
     *
     * @param value the value to add to the bias
     * @see Neuron#setBias(double)
     */
    public void addToBias(double value) {
        this.bias += value;
    }

    /**
     * Set the activation value to a new value.
     *
     * @param activationThreshold the new value for the activation value
     * @see Neuron#addToActivationValue(double)
     */
    public void setActivationThreshold(double activationThreshold) {
        this.activationThreshold = activationThreshold;
    }

    /**
     * Add a value to the activation value. To reduce the activation value, value may be negative.
     *
     * @param value the value to add to the activation value
     * @see Neuron#setActivationThreshold(double)
     */
    public void addToActivationValue(double value) {
        this.activationThreshold += value;
    }

    private double calcActivationThreshold() {
        return 0.0;
    }

    /**
     * Return the value of z = w^L * x + b, where b = bias
     *
     * @return the value of z
     */
    public double z(double[] x) {
        double sum = 0.0;
        for (int i = 0; i < this.weights.length; i++) {
            sum += this.weights[i] * x[i];
        }
        sum += this.bias;
        return sum;
    }

    /**
     * Return the value of the sigmoid function s = 1 / (1 + e^z).
     * Source: <a href="https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e">TowardsDataAnalysis</a>
     *
     * @param input the value of the input
     * @return the value of s
     * @see Neuron#z(double[])
     */
    private double sigmoidActivation(double[] input) {
        return (1 / (1 + Math.exp(-z(input))));
    }

    /**
     * Return the value of a linear function with z = w_T * x + b, where b = bias
     *
     * @param input the input value x
     * @return the value of z
     */
    private double linearActivation(double[] input) {
        return z(input);
    }

    public static double sigmoidDerivative(Neuron target, double[] input) {
        double activation = target.sigmoidActivation(input);
        return activation * (1 - activation);
    }

    /**
     * Indicates whether some other object is "equal to" this one.
     */
    @Override
    public boolean equals(Object obj) {
        return obj instanceof (Neuron n)
                && this.weights == n.weights
                && this.bias == n.bias
                && this.activationThreshold == n.activationThreshold;
    }

    /**
     * Returns a string representation of the object.
     *
     * @return a string representation of the object.
     */
    @Override
    public String toString() {
        return "N={" + Arrays.toString(this.weights) + ";b=" + this.bias + ";t=" + this.activationThreshold + "}";
    }
}

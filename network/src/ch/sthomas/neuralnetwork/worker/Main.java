package ch.sthomas.neuralnetwork.worker;

import ch.sthomas.neuralnetwork.Network;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[] inputs = new double[]{1.0, 3.5, 5, 0, 7.0, 0.25};
        Network n = new Network(5, 5, new int[]{10}, 5);
        System.out.println(
                Arrays.toString(
                        n.calculateOutputs(inputs)
                )
        );
        n.train(inputs, inputs);
    }
}
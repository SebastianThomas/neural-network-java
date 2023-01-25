package ch.sthomas.neuralnetwork.worker;

import ch.sthomas.neuralnetwork.MathUtils;
import ch.sthomas.neuralnetwork.Network;

import java.io.IOException;
import java.util.Arrays;
import java.util.Date;

public class Main {
    public static void main(String[] args) {
        trainNetworkToReturnSameValues();
    }

    public static void trainNetworkToReturnSameValues() {
        System.out.println(System.currentTimeMillis() + ": " + new Date());
        double[][] inputs = MathUtils.getRandomDoubleArrays(1000, 5);

        Network n = new Network(5, 5, new int[]{5, 5}, 5);

        System.out.println(formatInOutputs(new double[][]{inputs[0], inputs[1]}, new double[][]{n.calculateOutputs(inputs[0]), n.calculateOutputs(inputs[1])}));
        for (int i = 0; i < 1_000_000; i++) {
            n.train(inputs, inputs);
        }
        System.out.println("\n\n" + System.currentTimeMillis() + ": " + new Date());
        System.out.println(formatInOutputs(new double[][]{inputs[0], inputs[1]}, new double[][]{n.calculateOutputs(inputs[0]), n.calculateOutputs(inputs[1])}));

        try {
            n.saveToDisk("/tmp/networks/sameNumber");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void readTrainedNetworkToReturnSameValues() {
        double[][] inputs = MathUtils.getRandomDoubleArrays(1000, 5);
        try {
            Network n = Network.readFromDisk("/tmp/networks/sameNumber");
            System.out.println(Arrays.toString(inputs[0]) + "\t" + Arrays.toString(inputs[1]));
            System.out.println(Arrays.toString(n.calculateOutputs(inputs[0])) + "\t" + Arrays.toString(n.calculateOutputs(inputs[1])));
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static String formatInOutputs(double[][] inputs, double[][] outputs) {
        return "\n" +
                "Inputs: \t" + arrayToString(inputs[0]) + "\t" + arrayToString(inputs[1]) + "\n" +
                "Outputs:\t" + arrayToString(outputs[0]) + "\t" + arrayToString(outputs[1]);
    }

    public static String arrayToString(double[] input) {
        StringBuilder s = new StringBuilder("[");
        for (int i = 0; i < input.length; i++) {
            s.append(Math.round(input[i] * Math.pow(10, 7)) / Math.pow(10, 7));
            if (i + 1 < input.length) s.append(",\t");
        }
        s.append("]");
        return s.toString();
    }
}

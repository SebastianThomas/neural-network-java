package ch.sthomas.neuralnetwork.worker;

import ch.sthomas.neuralnetwork.MathUtils;
import ch.sthomas.neuralnetwork.Network;

import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        readTrainedNetworkToReturnSameValues();
    }

    public static void trainNetworkToReturnSameValues() {
        double[][] inputs = MathUtils.getRandomDoubleArrays(1000, 5);

        Network n = new Network(5, 5, new int[]{}, 5);
        System.out.println(Arrays.toString(n.calculateOutputs(inputs[0])));
        for (int i = 0; i < 10_000; i++) {
            n.train(inputs, inputs);
            System.out.println();
            System.out.println(Arrays.toString(inputs[0]) + "\t" + Arrays.toString(inputs[1]));
            System.out.println(Arrays.toString(n.calculateOutputs(inputs[0])) + "\t" + Arrays.toString(n.calculateOutputs(inputs[1])));
        }
        System.out.println(n);

        try {
            n.saveToDisk("/tmp/networks/sameNumber_1000_10000");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void readTrainedNetworkToReturnSameValues() {
        double[][] inputs = MathUtils.getRandomDoubleArrays(1000, 5);
        try {
            Network n = Network.readFromDisk("/tmp/networks/sameNumber_1000_10000");
            System.out.println(Arrays.toString(inputs[0]) + "\t" + Arrays.toString(inputs[1]));
            System.out.println(Arrays.toString(n.calculateOutputs(inputs[0])) + "\t" + Arrays.toString(n.calculateOutputs(inputs[1])));
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}

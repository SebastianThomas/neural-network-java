package ch.sthomas.neuralnetwork.worker;

import ch.sthomas.neuralnetwork.MathUtils;
import ch.sthomas.neuralnetwork.Network;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.Date;

public class Main {
    public static void main(String[] args) {
//        trainNetworkToReturnSameValues();
//        readTrainedNetworkToReturnSameValues();
//        trainNetworkToReturnOR();
    }

    public static void trainNetworkToReturnSameValues() {
        System.out.println(System.currentTimeMillis() + ": " + new Date());
        long now = System.currentTimeMillis();
        double[][] inputs = MathUtils.getRandomDoubleArrays(1000, 5);

        Network n = new Network(5, 5, new int[]{5}, 5);
        try {
            n.saveToDisk("/tmp/networks/sameNumber/" + now + "/untrained");
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(formatInOutputs(new double[][]{inputs[0], inputs[1]}, new double[][]{n.calculateOutputs(inputs[0]), n.calculateOutputs(inputs[1])}));
        for (int i = 0; i < 10_000; i++) {
            n.train(inputs, inputs);
        }
        System.out.println("\n\n" + System.currentTimeMillis() + ": " + new Date());
        System.out.println(formatInOutputs(new double[][]{inputs[0], inputs[1]}, new double[][]{n.calculateOutputs(inputs[0]), n.calculateOutputs(inputs[1])}));

        try {
            n.saveToDisk("/tmp/networks/sameNumber/" + now + "/trained");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void trainNetworkToReturnOR() {
        Instant start = Instant.now();

        double[][] inputs = MathUtils.getRandomBooleanArrays(1000, 2);
        double[][] expected = new double[inputs.length][2];
        for (int i = 0; i < expected.length; i++) {
            expected[i][0] = Double.compare(inputs[i][0], 1) == 0 || Double.compare(inputs[i][1], 1) == 0
                    ? 1.0
                    : 0.0;
        }

        Network n = new Network(2, 2, new int[]{2}, 1);
        try {
            n.saveToDisk("/tmp/networks/OR/" + start.getNano() + "/untrained");
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(formatInOutputs(new double[][]{inputs[0], inputs[1]}, new double[][]{n.calculateOutputs(inputs[0]), n.calculateOutputs(inputs[1])}));

        for (int i = 0; i < 10_000; i++) n.train(inputs, expected);

        System.out.println(formatInOutputs(new double[][]{inputs[0], inputs[1]}, new double[][]{n.calculateOutputs(inputs[0]), n.calculateOutputs(inputs[1])}));

        Instant end = Instant.now();
        Duration d = Duration.between(start, end);
        System.out.format("%dD, %02d:%02d:%02d.%04d \n", d.toDays(),
                d.toHours(), d.toMinutes(), d.getSeconds(), d.toMillis());
    }

    public static void readTrainedNetworkToReturnSameValues(String number) {
        double[][] inputs = MathUtils.getRandomDoubleArrays(1000, 5);
        try {
            Network n = Network.readFromDisk("/tmp/networks/sameNumber/" + number + "/trained");
            System.out.println(formatInOutputs(new double[][]{inputs[0], inputs[1]}, new double[][]{n.calculateOutputs(inputs[0]), n.calculateOutputs(inputs[1])}));
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static String formatInOutputs(double[][] inputs, double[][] outputs) {
        return "\n" + "Inputs: \t" + arrayToString(inputs[0]) + "\t" + arrayToString(inputs[1]) + "\n" + "Outputs:\t" + arrayToString(outputs[0]) + "\t" + arrayToString(outputs[1]);
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

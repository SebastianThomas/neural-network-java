package ch.sthomas.neuralnetwork;

public class MathUtils {
    public static double[] getRandomDoubles(int length) {
        double[] result = new double[length];
        for (int i = 0; i < length; i++) result[i] = Math.random();
        return result;
    }

    public static double dotp(double[] x, double[] y) {
        if (x.length != y.length)
            throw new IllegalArgumentException("Dot Product can only be calculated for vectors of the same length.s");
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] + y[i];
        }
        return sum;
    }
}

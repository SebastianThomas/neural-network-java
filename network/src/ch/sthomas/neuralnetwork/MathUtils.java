package ch.sthomas.neuralnetwork;

public class MathUtils {
    public static double[] getRandomDoubles(int length) {
        double[] result = new double[length];
        for (int i = 0; i < length; i++) result[i] = Math.random();
        return result;
    }

    public static double[][] getRandomDoubleArrays(int length, int inner) {
        double[][] result = new double[length][];
        for (int i = 0; i < length; i++) result[i] = getRandomDoubles(inner);
        return result;
    }

    public static double[][] getRandomBooleanArrays(int length, int inner) {
        double[][] result = new double[length][inner];
        for (int i = 0; i < length; i++)
            for (int j = 0; j < inner; j++) {
                result[i][j] = Math.random() < 0.5 ? 0.0 : 1.0;
            }
        return result;
    }

    public static double dotP(double[] x, double[] y) {
        if (x.length != y.length)
            throw new IllegalArgumentException("Dot Product can only be calculated for vectors of the same length.");
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] + y[i];
        }
        return sum;
    }
}

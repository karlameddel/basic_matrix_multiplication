import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;

/** Basic triple-loop (i-j-k) dense matrix multiplication, double precision.
 *  Usage: java MatMul A.bin B.bin n reps warmup [out.bin]
 *  Timed region: kernel only. Output: rep,seconds,checksum */
public class MatMul {
    static double[] load(String path, int n) throws IOException {
        double[] m = new double[n * n];
        try (FileChannel ch = FileChannel.open(Paths.get(path), StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocate(n * n * 8).order(ByteOrder.LITTLE_ENDIAN);
            while (buf.hasRemaining()) if (ch.read(buf) < 0) break;
            buf.flip();
            buf.asDoubleBuffer().get(m);
        }
        return m;
    }

    static void matmul(double[] A, double[] B, double[] C, int n) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++)
                    s += A[i * n + k] * B[k * n + j];
                C[i * n + j] = s;
            }
    }

    public static void main(String[] args) throws IOException {
        int n = Integer.parseInt(args[2]);
        int reps = Integer.parseInt(args[3]), warmup = Integer.parseInt(args[4]);
        double[] A = load(args[0], n), B = load(args[1], n);
        double[] C = new double[n * n];

        for (int r = -warmup; r < reps; r++) {
            java.util.Arrays.fill(C, 0.0);
            long t0 = System.nanoTime();
            matmul(A, B, C, n);
            long t1 = System.nanoTime();
            double checksum = 0.0;
            for (double v : C) checksum += v;
            if (r >= 0) System.out.printf(java.util.Locale.ROOT, "%d,%.9f,%.12e%n", r, (t1 - t0) / 1e9, checksum);
        }

        if (args.length >= 6) {
            ByteBuffer out = ByteBuffer.allocate(n * n * 8).order(ByteOrder.LITTLE_ENDIAN);
            out.asDoubleBuffer().put(C);
            Files.write(Paths.get(args[5]), out.array());
        }
    }
}
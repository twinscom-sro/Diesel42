package environment;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Utilities {

    public static void writeFile(String outFile, StringBuilder sb) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
            writer.write( sb.toString() );
            //writer.write("\n");
            writer.close();
            System.out.println("Utilities::Writing file " + outFile);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

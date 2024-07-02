import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class ByteStreamsExample {
    public static void main(String[] args) throws IOException {
        FileInputStream inStream = null;
        FileOutputStream outStream = null;
        
        try {
            inStream = new FileInputStream("D:\\Ravi\\2023\\Interview\\System-design-interview\\JavaInterview\\Java IO Streams\\source.txt");
            outStream = new FileOutputStream("D:\\Ravi\\2023\\Interview\\System-design-interview\\JavaInterview\\Java IO Streams\\destination.txt");
            int content;
            while ((content = inStream.read()) != -1) {
                outStream.write((byte) content);
            }
        }
        finally {
            if (inStream != null)
                inStream.close();
            if (outStream != null)
                outStream.close();
        }
    }
}
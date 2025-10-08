package environment;

import java.io.IOException;

import org.apache.http.Header;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

public class HttpClient {

    private CloseableHttpClient httpClient;

    HttpClient(){
        httpClient = HttpClients.createDefault();
    }

    public String get(String url ){

        HttpClient obj = new HttpClient();
        String output="";

        try {
            output = obj.sendGet(url);
            obj.close();
        }catch( Exception e ) {

        } finally {
        }

        return output;
    }


    void close() throws IOException {
        httpClient.close();
    }

    String sendGet( String url ) throws Exception {
        String result = "";

        // https://www.fmlabs.com/reference/default.htm?url=WeightedMA.htm
        // https://www.alphavantage.co/documentation/

        HttpGet request = new HttpGet( url );
        //"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&datatype=csv&symbol=IBM&apikey=WKOBZ8KHQFU2125K"
        //"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=WKOBZ8KHQFU2125K"
        //"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=IBM&apikey=WKOBZ8KHQFU2125K"
        //"https://www.alphavantage.co/query?function=MACD&symbol=IBM&interval=daily&series_type=open&apikey=WKOBZ8KHQFU2125K"
        //"https://www.alphavantage.co/query?function=STOCHF&symbol=IBM&interval=daily&apikey=WKOBZ8KHQFU2125K"

        try (CloseableHttpResponse response = httpClient.execute(request)) {

            // Get HttpResponse Status
            System.out.println(response.getStatusLine().toString());

            HttpEntity entity = response.getEntity();
            Header headers = entity.getContentType();
            System.out.println(headers);

            if (entity != null) {
                // return it as a String
                result = EntityUtils.toString(entity);
            }

        }

        return( result );
    }

}

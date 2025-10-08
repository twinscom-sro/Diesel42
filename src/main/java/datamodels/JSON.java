package datamodels;

import org.bson.Document;

public class JSON {

    String message;
    Document content;

    public JSON(String _message) {
        message = _message;
    }
    public JSON(String _message, Document _doc){
        message = _message;
        content = _doc;
    }

    public static JSON ERROR(String _message) {
        return new JSON(_message);
    }

    public static JSON DATA(Document doc) {
        return new JSON("OK", doc);
    }
}

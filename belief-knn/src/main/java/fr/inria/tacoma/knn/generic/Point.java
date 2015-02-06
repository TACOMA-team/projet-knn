package fr.inria.tacoma.knn.generic;

import com.fasterxml.jackson.annotation.JsonProperty;
import fr.inria.tacoma.knn.LabelledPoint;

public class Point<T> extends LabelledPoint {

    private T value;


    public Point(@JsonProperty("sensor") String sensor,
                 @JsonProperty("label") String label,
                 @JsonProperty("timestamp") double timestamp,
                 T value) {
        super(sensor, label, timestamp);
        this.value = value;
    }

    public T getValue() {
        return value;
    }
}

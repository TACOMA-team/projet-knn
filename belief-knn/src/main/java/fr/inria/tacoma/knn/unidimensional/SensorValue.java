package fr.inria.tacoma.knn.unidimensional;

import com.fasterxml.jackson.annotation.JsonProperty;
import fr.inria.tacoma.knn.generic.Point;

public class SensorValue extends Point<Double> {

    public SensorValue(@JsonProperty("sensor") String sensor,
                       @JsonProperty("label") String label,
                       @JsonProperty("timestamp") double timestamp,
                       @JsonProperty("value") Double value) {
        super(sensor, label, timestamp, value);
    }

    @Override
    public String toString() {
        return "{" +
                "\"value\" : " + value +
                ", \"sensor\" : \"" + sensor + '\"' +
                ", \"label\" : \"" + label + '\"' +
                ", \"timestamp\" : " + timestamp +
                '}';
    }
}

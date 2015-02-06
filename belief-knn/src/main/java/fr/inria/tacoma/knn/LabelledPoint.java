package fr.inria.tacoma.knn;

import com.fasterxml.jackson.annotation.JsonProperty;

public class LabelledPoint {
    protected String sensor;
    protected String label;
    protected double timestamp;

    public LabelledPoint(@JsonProperty("sensor") String sensor, @JsonProperty("label") String label,
                         @JsonProperty("timestamp")double timestamp) {
        this.sensor = sensor;
        this.label = label;
        this.timestamp = timestamp;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getSensor() {
        return sensor;
    }

    public void setSensor(String sensor) {
        this.sensor = sensor;
    }

    public double getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(double timestamp) {
        this.timestamp = timestamp;
    }
}

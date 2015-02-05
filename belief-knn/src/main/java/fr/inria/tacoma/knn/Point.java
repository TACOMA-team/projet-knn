package fr.inria.tacoma.knn;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Point {
    private double value;
    private String sensor;
    private String label;
    private double timestamp;

    public Point(@JsonProperty("value") double value,@JsonProperty("label")  String label) {
        this.value = value;
        this.label = label;
    }

    public double getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public void setValue(double value) {
        this.value = value;
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

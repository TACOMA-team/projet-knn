package fr.inria.tacoma.knn;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Point {
    private double value;
    private String label;

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

    @Override
    public String toString() {
        return "{" +
                "\"value\" : " + value +
                ", \"label\" : \"" + label + '\"' +
                '}';
    }
}

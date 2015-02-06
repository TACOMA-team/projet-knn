package fr.inria.tacoma.knn.bidimensional;

public class Coordinate {
    private double x;
    private double y;

    public Coordinate(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double distance(Coordinate that) {
        double xDiff = this.getX() - that.getX();
        double yDiff = (this.getY() - that.getY());
        return Math.sqrt(xDiff * xDiff + yDiff * yDiff);
    }
}

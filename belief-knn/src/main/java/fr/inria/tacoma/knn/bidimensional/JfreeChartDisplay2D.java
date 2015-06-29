package fr.inria.tacoma.knn.bidimensional;


import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.GrayPaintScale;
import org.jfree.chart.renderer.PaintScale;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.data.xy.AbstractXYZDataset;
import org.jfree.data.xy.XYZDataset;

import java.awt.*;
import java.util.ArrayList;

public class JfreeChartDisplay2D {

    public static ChartPanel getChartPanel(SensorBeliefModel<Coordinate> beliefModel,
                                           int numPointsX, double maxX, int numPointsY, double maxY,
                                           StateSet stateSet) {
        XYZDataset dataset = new BeliefDataSet(beliefModel, stateSet, numPointsX, numPointsY, maxX,
                maxY);
        NumberAxis xAxis = new NumberAxis("X");
        xAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        xAxis.setLowerMargin(0.0);
        xAxis.setUpperMargin(0.0);
        NumberAxis yAxis = new NumberAxis("Y");
        yAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        yAxis.setLowerMargin(0.0);
        yAxis.setUpperMargin(0.0);
        XYBlockRenderer renderer = new XYBlockRenderer();
        renderer.setBlockWidth(maxX / numPointsX);
        renderer.setBlockHeight(maxY / numPointsY);
        PaintScale scale = new GrayPaintScale(0.0, 1.0);
        renderer.setPaintScale(scale);
        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);
        plot.setDomainGridlinesVisible(false);
        plot.setRangeGridlinePaint(Color.white);
        JFreeChart chart = new JFreeChart("value for " + stateSet, plot);
        chart.removeLegend();
        chart.setBackgroundPaint(Color.white);
        return new ChartPanel(chart);
    }

    private static class BeliefDataSet extends AbstractXYZDataset {


        private int numX;
        private int numY;
        ArrayList<Double> values;
        private final double xStep;
        private final double yStep;

        public BeliefDataSet(SensorBeliefModel<Coordinate> beliefModel,
                             StateSet stateSet, int numX, int numY, double rangeX, double rangeY) {
            this.numX = numX;
            this.numY = numY;
            this.values = new ArrayList<>();

            xStep = rangeX / numX;
            yStep = rangeY / numY;
            for (int x = 0; x < numX; x++) {
                for (int y = 0; y < numY; y++) {
                    values.add(beliefModel.toMass(new Coordinate(x * xStep, y * yStep)).get(
                            stateSet));
                }
            }
        }

        @Override
        public int getSeriesCount() {
            return 1;
        }

        @Override
        public Comparable getSeriesKey(int series) {
            return "test";
        }

        @Override
        public Number getZ(int series, int item) {
            return values.get(item);
        }

        @Override
        public int getItemCount(int series) {
            return numX * numY;
        }

        @Override
        public Number getX(int series, int item) {
            return ((double)(item % numX) * xStep);
        }

        @Override
        public Number getY(int series, int item) {
            return (double)(item / numX) * yStep;
        }
    }
}

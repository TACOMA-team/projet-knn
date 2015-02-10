package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.DefaultXYDataset;

import java.util.*;
import java.util.stream.IntStream;

public class JfreeChartDisplay1D {

    public static ChartPanel getChartPanel(SortedMap<Double, MassFunction> masses,
                                           FrameOfDiscernment frame) {
        HashMap<StateSet,double[][]> dataMap = new HashMap<>();
        for (int i = 1; i <= frame.card(); i++) {
            for (StateSet stateSet : frame.getStateSetsWithCard(i)) {
                dataMap.put(stateSet,
                        new double[][]{sensorValueArray(masses), new double[masses.size()]});
            }
        }

        int index = 0;
        for (Map.Entry<Double, MassFunction> doubleMassEntry : masses.entrySet()) {
            final int dataSetIndex = index; //need a final variable to use in lambda expression
            MassFunction mass = doubleMassEntry.getValue();
            mass.foreachFocalElement((stateSet, massValue) ->  {
                double[][] dataArray = dataMap.get(stateSet);
                dataArray[1][dataSetIndex] = massValue;
            });
            index++;
        }
        DefaultXYDataset dataSet = new DefaultXYDataset();
        for (Map.Entry<StateSet, double[][]> stateSetEntry : dataMap.entrySet()) {
            dataSet.addSeries(stateSetEntry.getKey().toString(), stateSetEntry.getValue());
        }
        JFreeChart chart = ChartFactory.createXYLineChart("sensor to belief mapping", "sensor value",
                "mass", dataSet);
        ChartPanel chartPanel = new ChartPanel(chart);
        ((XYPlot)chart.getPlot()).getRangeAxis().setRange(0,1);
        return chartPanel;
    }

    public static ChartPanel getChartPanel(SensorBeliefModel<Double> beliefModel, int numPoints, double min,
                                           double max) {
        TreeMap<Double, MassFunction> massFunctionSet = new TreeMap<>();
        Map<Double, MassFunction> syncMap = Collections.synchronizedMap(massFunctionSet);
        IntStream.range(0, numPoints).parallel()
                .mapToDouble(x -> min + (x * ((max-min) / numPoints)))
                .forEach(value -> syncMap.put(value, beliefModel.toMass(value)));

        return getChartPanel(massFunctionSet, beliefModel.getFrame());
    }

    static private double[] sensorValueArray(Map<Double, MassFunction> masses) {
        int index = 0;
        double[] array = new double[masses.size()];
        for (Double sensorValue : masses.keySet()) {
            array[index] = sensorValue;
            index++;
        }
        return array;
    }
}

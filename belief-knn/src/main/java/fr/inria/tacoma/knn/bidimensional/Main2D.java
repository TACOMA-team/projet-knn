package fr.inria.tacoma.knn.bidimensional;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.util.Mass;
import fr.inria.tacoma.knn.TrainingSet;
import fr.inria.tacoma.knn.generic.KnnBelief;
import fr.inria.tacoma.knn.generic.Point;
import fr.inria.tacoma.knn.unidimensional.SensorValue;
import org.jfree.chart.ChartPanel;

import javax.swing.*;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class Main2D {

    public static final double ALPHA = 0.6;

    public static void main(String[] args) throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");
        List<SensorValue> points = getPoints("absence", "absence-motion1.json");
        points.addAll(getPoints("presence", "presence-motion1.json"));
        points.sort((p1, p2) -> Double.compare(p1.getTimestamp(), p1.getTimestamp()));
//        points.forEach(p -> p.setValue(Math.abs(p.getValue() - 2048)));
        TrainingSet trainingSet = new TrainingSet(points);
        List<Point<Coordinate>> points2D = to2D(points);

        //showBestMatch(frame, points2D);
        show(frame.toStateSet("absence", "presence"), new KnnBelief<Coordinate>(points2D, 12, ALPHA, frame,
                Main2D::optimizedDuboisAndPrade, Coordinate::distance));
    }

    /**
     * Shows the model having the lowest error depending on K.
     *  @param frame       frame of discernment
     * @param points training set on which we apply knn.
     */
    private static void showBestMatch(StateSet stateSet, FrameOfDiscernment frame, List<Point<Coordinate>> points) {
        show(stateSet, getBestKnnBelief(frame, points));

    }

    private static void show(StateSet stateSet, KnnBelief<Coordinate> model) {
        ChartPanel chartPanel = JfreeChartDisplay2D.getChartPanel(model, 250, 2000.0,
                250, 2000.0, stateSet);
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        windowFrame.setContentPane(chartPanel);
        windowFrame.pack();
        windowFrame.setVisible(true);
    }

    private static KnnBelief<Coordinate> getBestKnnBelief(FrameOfDiscernment frame,
                                              List<Point<Coordinate>> points) {
        return getBestKnnBelief(frame, points, points.size() - 1);
    }

    /**
     * Finds the model having the lowest error depending on K. This iterate the knn algorithm by
     * incrementing k and calculating the error. It then return the model with the minimum error.
     *
     * @param frame            frame of discernment
     * @param points      training set to use
     * @param maxNeighborCount maximum to use for k (the effective max will be limited by the size
     *                         of the training set)
     * @return
     */
    private static KnnBelief<Coordinate> getBestKnnBelief(FrameOfDiscernment frame,
                                                          List<Point<Coordinate>> points,
                                                          int maxNeighborCount) {
        double lowestError = Double.POSITIVE_INFINITY;
        KnnBelief<Coordinate> bestModel = null;

        maxNeighborCount = Math.min(maxNeighborCount, points.size() - 1);
        for (int neighborCount = 1; neighborCount <= maxNeighborCount; neighborCount++) {
            KnnBelief<Coordinate> beliefModel =
                    new KnnBelief<>(points, neighborCount, ALPHA, frame,
                    Main2D::optimizedDuboisAndPrade, Coordinate::distance);
            double error = error(points, beliefModel);
            if (error < lowestError) {
                lowestError = error;
                bestModel = beliefModel;
            }
        }
        System.out.println("lowest error: " + lowestError);
        System.out.println("bestNeighborCount: " + bestModel.getK());
        return bestModel;
    }


    private static double error(List<Point<Coordinate>> points, SensorBeliefModel<Coordinate> model) {
        return points.stream().mapToDouble(point -> {
            MassFunction actualMassFunction = model.toMass(point.getValue());
            MassFunction idealMassFunction = new MassFunctionImpl(model.getFrame());
            idealMassFunction.set(model.getFrame().toStateSet(point.getLabel()), 1);
            idealMassFunction.putRemainingOnIgnorance();
            double distance = Mass.jousselmeDistance(actualMassFunction, idealMassFunction);
            return distance * distance;
        }).sum();
    }


    private static List<Point<Coordinate>> to2D(List<SensorValue> sortedPoints) {
        List<Point<Coordinate>> points = new ArrayList<>(sortedPoints.size() - 1);
        for (int i = 1; i < sortedPoints.size(); i++) {
            SensorValue point = sortedPoints.get(i);
            SensorValue prevPoint = sortedPoints.get(i - 1);
            double absoluteDerivative = Math.abs(point.getValue() - prevPoint.getValue())
                    / (point.getTimestamp() - prevPoint.getTimestamp());
            Coordinate coord = new Coordinate(Math.abs(point.getValue() - 2048), absoluteDerivative);
            points.add(new Point<>(point.getSensor(), point.getLabel(), point.getTimestamp(),
                    coord));
        }
        return points;
    }

    /**
     * An hybrid fusion mecanism which apply dempster for every points with the same label, end the
     * fuse the resulting mass functions with dubois and prade. This allow to perform a very
     * efficient dubois and prade.
     *
     * @param masses masses to fuse
     * @return fused mass function
     */
    private static MassFunction optimizedDuboisAndPrade(List<MassFunction> masses) {
        List<MassFunction> optimizedMasses = new ArrayList<>(masses);
        for (int refMassIndex = 0; refMassIndex < optimizedMasses.size(); refMassIndex++) {
            MassFunction referenceMass = optimizedMasses.get(refMassIndex);
            for (int j = refMassIndex + 1; j < optimizedMasses.size(); ) {
                MassFunction mass2 = optimizedMasses.get(j);
                if (referenceMass.getFocalStateSets().equals(mass2.getFocalStateSets())) {
                    referenceMass = Combinations.dempster(referenceMass, mass2);
                    optimizedMasses.remove(j);
                } else {
                    j++;
                }
            }
            optimizedMasses.set(refMassIndex, referenceMass);
        }
        return Combinations.duboisAndPrade(optimizedMasses);
    }

    /**
     * Parse the given file to extract points. One file is expected to contain only one sensor.
     *
     * @param label state in which the sample was take (i.e. presence or absence)
     * @param file  path to the file
     * @return list of points in the file
     * @throws java.io.IOException
     */
    private static List<SensorValue> getPoints(String label, String file) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        InputStream resourceAsStream = Thread.currentThread().getContextClassLoader()
                .getResourceAsStream(file);
        MappingIterator<SensorValue> iterator = mapper.readValues(
                new JsonFactory().createParser(resourceAsStream),
                SensorValue.class);
        List<SensorValue> points = new ArrayList<>();
        while (iterator.hasNext()) {
            SensorValue next = iterator.next();
            next.setLabel(label);
            points.add(next);
        }
        return points;
    }

}

package fr.inria.tacoma.knn.unidimensional;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.util.Mass;
import fr.inria.tacoma.knn.generic.KnnBelief;
import fr.inria.tacoma.knn.util.KnnUtils;
import org.jfree.chart.ChartPanel;

import javax.swing.*;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class Main1D {

    public static final double ALPHA = 0.2;

    public static void main(String[] args) throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");
        List<SensorValue> points = getPoints("absence", "absence-motion1.json");
        points.addAll(getPoints("presence", "presence-motion1.json"));
        points.sort((p1, p2) -> Double.compare(p1.getValue(), p2.getValue()));
        points.forEach(p -> p.setValue(Math.abs(p.getValue() - 2048)));
        showBestMatch(frame, points);
//        displayTabsDependingOnK(frame, trainingSet);
//        displayWithWeakening(frame, trainingSet, 0, 2000);
    }

//    /**
//     * Display the resulting model when we apply knn and then our weakening algorithm.
//     *  @param frame       frame of discernment on which we are working
//     * @param trainingSet training set used to apply knn
//     * @param min         minimum sensor value
//     * @param max         maximum sensor value
//     */
//    private static void displayWithWeakening(FrameOfDiscernment frame, List<SensorValue> trainingSet,
//                                             double min, double max) {
//        KnnBelief<Double> beliefModel = getBestKnnBelief(frame, trainingSet);
//        DiscountingBeliefModel weakened = generateWeakeningModel(beliefModel, trainingSet);
//
//        ChartPanel chartPanel = JfreeChartDisplay.getChartPanel(weakened, 2000, min, max);
//        JFrame windowFrame = new JFrame();
//        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
//        windowFrame.setContentPane(chartPanel);
//        windowFrame.pack();
//        windowFrame.setVisible(true);
//    }

    private static void displayTabsDependingOnK(FrameOfDiscernment frame,
                                                List<SensorValue> trainingSet) {
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        JTabbedPane tabbedPane = new JTabbedPane();

        windowFrame.setContentPane(tabbedPane);
        windowFrame.setSize(800, 500);
        windowFrame.setVisible(true);


        for (int neighborCount = 1; neighborCount <= 120; neighborCount++) {
            KnnBelief<Double>  beliefModel = new KnnBelief<>(trainingSet, neighborCount, ALPHA, frame,
                    KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b));

            JPanel panel = JfreeChartDisplay.getChartPanel(beliefModel, 2000, 0, 2000);
            tabbedPane.addTab("" + neighborCount, panel);
        }
    }



    /**
     * Parse the given file to extract points. One file is expected to contain only one sensor.
     *
     * @param label state in which the sample was take (i.e. presence or absence)
     * @param file  path to the file
     * @return list of points in the file
     * @throws IOException
     */
    private static List<SensorValue> getPoints(String label, String file) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, true);
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

    /**
     * Shows the model having the lowest error depending on K.
     *  @param frame       frame of discernment
     * @param trainingSet training set on which we apply knn.
     */
    private static void showBestMatch(FrameOfDiscernment frame, List<SensorValue> trainingSet) {
        KnnBelief<Double> bestModel = getBestKnnBelief(frame, trainingSet);
        show(bestModel);
    }

    private static void show(KnnBelief<Double> model) {
        ChartPanel chartPanel = JfreeChartDisplay.getChartPanel(model, 2000, 0, 2000);
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        windowFrame.setContentPane(chartPanel);
        windowFrame.pack();
        windowFrame.setVisible(true);
    }

    private static KnnBelief<Double> getBestKnnBelief(FrameOfDiscernment frame,
                                                      List<SensorValue> trainingSet) {
        return getBestKnnBelief(frame, trainingSet, trainingSet.size() - 1);
    }

    /**
     * Finds the model having the lowest error depending on K. This iterate the knn algorithm by
     * incrementing k and calculating the error. It then return the model with the minimum error.
     *
     * @param frame            frame of discernment
     * @param trainingSet      training set to use
     * @param maxNeighborCount maximum to use for k (the effective max will be limited by the size
     *                         of the training set)
     * @return
     */
    private static KnnBelief<Double> getBestKnnBelief(FrameOfDiscernment frame, List<SensorValue> trainingSet,
                                              int maxNeighborCount) {
        double lowestError = Double.POSITIVE_INFINITY;
        KnnBelief<Double> bestModel = null;

        maxNeighborCount = Math.min(maxNeighborCount, trainingSet.size() - 1);
        for (int neighborCount = 1; neighborCount <= maxNeighborCount; neighborCount++) {
            KnnBelief<Double> beliefModel = new KnnBelief<>(trainingSet,
                    neighborCount, ALPHA, frame, KnnUtils::optimizedDuboisAndPrade,
                    (a,b) -> Math.abs(a - b));
            double error = error(trainingSet, beliefModel);
            if (error < lowestError) {
                lowestError = error;
                bestModel = beliefModel;
            }
        }
        System.out.println("lowest error: " + lowestError);
        System.out.println("bestNeighborCount: " + bestModel.getK());
        return bestModel;
    }


    private static double error(List<SensorValue> points, SensorBeliefModel model) {
        return points.stream().mapToDouble(point -> {
            MassFunction actualMassFunction = model.toMass(point.getValue());
            MassFunction idealMassFunction = new MassFunctionImpl(model.getFrame());
            idealMassFunction.set(model.getFrame().toStateSet(point.getLabel()), 1);
            idealMassFunction.putRemainingOnIgnorance();
            double distance = Mass.jousselmeDistance(actualMassFunction, idealMassFunction);
            return distance * distance;
        }).sum();
    }

//    private static DiscountingBeliefModel generateWeakeningModel(SensorBeliefModel model,
//                                                                 List<SensorValue> trainingSet) {
//        DecisionStrategy decisionStrategy =
//                new CriteriaDecisionStrategy(0.5, 0.6, 0.7, Criteria::betP);
//        WeakeningFunction weakeningFunction = new WeakeningFunction();
//        DiscountingBeliefModel discountingBeliefModel =
//                new DiscountingBeliefModel(model, weakeningFunction);
//        trainingSet.stream().forEach(point -> {
//
//            Decision decision = decisionStrategy.decide(
//                    discountingBeliefModel.toMass(point.getValue()));
//            StateSet actualDecision = decision.getStateSet();
//            StateSet expectedDecision = model.getFrame().toStateSet(point.getLabel());
//
//            if (!actualDecision.includesOrEquals(expectedDecision)) {
//                System.out.println("bad decision for value: " + point.getValue());
//                double weakening = decision.getConfidence() - 0.5;
//                weakeningFunction.addWeakeningPoint(point.getValue(), weakening,
//                        trainingSet.getStandardDevs().get(point.getLabel()));
//            }
//        });
//
//
//        return discountingBeliefModel;
//    }

    private static class WeakeningFunction implements Function<Double, Double> {
        private Map<Double, Double> alphas = new HashMap<>();
        private Map<Double, Double> stdDevs = new HashMap<>();

        public void addWeakeningPoint(double center, double weakening, double amplitude) {
            alphas.put(center, weakening);
            stdDevs.put(center, amplitude);
        }

        @Override
        public Double apply(Double value) {
            return alphas.entrySet().stream().mapToDouble(entry -> {
                Double max = entry.getValue();
                double diff = entry.getKey() - value;
                return max * Math.exp(-Math.pow(3 * diff / stdDevs.get(entry.getKey()), 2));
            }).sum();
        }
    }

}

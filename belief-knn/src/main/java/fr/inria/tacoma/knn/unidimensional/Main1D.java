package fr.inria.tacoma.knn.unidimensional;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.knn.core.KnnBelief;
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

    public static final double ALPHA = 0.3;
    public static final double TRAINING_SET_RATIO = 0.6;
    public static final int SENSOR_VALUE_CENTER = 2048;

    public static void main(String[] args) throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");

        List<SensorValue> absence = getPoints("absence", "absence-motion1.json");
        List<SensorValue> presence = getPoints("presence", "presence-motion1.json");

        // extracting cross validation data
        List<SensorValue> crossValidation = KnnUtils.extractSubList(absence, TRAINING_SET_RATIO);
        crossValidation.addAll(KnnUtils.extractSubList(presence, TRAINING_SET_RATIO));
        crossValidation.forEach(p -> p.setValue(Math.abs(p.getValue() - SENSOR_VALUE_CENTER)));

        //creating training set
        List<SensorValue> trainingSet = new ArrayList<>();
        trainingSet.addAll(absence);
        trainingSet.addAll(presence);
        trainingSet.forEach(p -> p.setValue(Math.abs(p.getValue() - SENSOR_VALUE_CENTER)));

        showBestMatch(frame, trainingSet, crossValidation);
//        displayTabsDependingOnK(frame, trainingSet);
    }

    private static void displayTabsDependingOnK(FrameOfDiscernment frame,
                                                List<SensorValue> trainingSet) {
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        JTabbedPane tabbedPane = new JTabbedPane();

        windowFrame.setContentPane(tabbedPane);
        windowFrame.setSize(800, 500);
        windowFrame.setVisible(true);


        for (int neighborCount = 1; neighborCount <= trainingSet.size(); neighborCount++) {
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
     * @param frame       frame of discernment
     * @param trainingSet training set on which we apply knn.
     * @param crossValidation set of points used for cross validation
     */
    private static void showBestMatch(FrameOfDiscernment frame, List<SensorValue> trainingSet,
                                      List<SensorValue> crossValidation) {
        KnnBelief<Double> bestModel = KnnUtils.getBestKnnBelief(frame, trainingSet, crossValidation,
                ALPHA, (a,b) -> Math.abs(a - b));
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


// FIXME rewrite this function
//    private static DiscountingBeliefModel generateWeakeningModel(SensorBeliefModel<Double> model,
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
//                //FIXME This weakening is arbitrary
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

package fr.inria.tacoma.knn.unidimensional;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.criteria.Criteria;
import fr.inria.tacoma.bft.decision.CriteriaDecisionStrategy;
import fr.inria.tacoma.bft.decision.Decision;
import fr.inria.tacoma.bft.decision.DecisionStrategy;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.util.ConsonantBeliefModel;
import fr.inria.tacoma.knn.util.DiscountingBeliefModel;
import fr.inria.tacoma.knn.util.KnnUtils;
import org.jfree.chart.ChartPanel;

import javax.swing.*;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Main1D {

    public static final double ALPHA = 0.2;
    public static final double TRAINING_SET_RATIO = 0.6;
    public static final int SENSOR_VALUE_CENTER = 2048;

    public static void main(String[] args) throws IOException {
        testKfold(3);
        testKfold(3);
        testKfold(3);
//        printAbsenceAndPresenceDependingOnAlpha();
        testNormal();
    }

    private static void testNormal() throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");

        String absenceFile = "samples/sample-1/sensor-1/absence-motion1.json";
        String presenceFile = "samples/sample-1/sensor-1/presence-motion1.json";
        List<SensorValue> absence = getPoints("absence", absenceFile);
        List<SensorValue> presence = getPoints("presence", presenceFile);

        System.out.println(
                "using " + absenceFile + " for absence and " + presenceFile + " for presence");

        absence.forEach(p -> p.setValue(Math.abs(p.getValue() - SENSOR_VALUE_CENTER)));
        presence.forEach(p -> p.setValue(Math.abs(p.getValue() - SENSOR_VALUE_CENTER)));
        crossValidation(frame, absence, presence);
    }

    private static void testKfold(int k) throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");

        String absenceFile = "samples/sample-1/sensor-1/absence-motion1.json";
        String presenceFile = "samples/sample-1/sensor-1/presence-motion1.json";
        List<SensorValue> absence = getPoints("absence", absenceFile);
        List<SensorValue> presence = getPoints("presence", presenceFile);

        System.out.println(
                "using " + absenceFile + " for absence and " + presenceFile + " for presence");
        absence.forEach(p -> p.setValue(Math.abs(p.getValue() - SENSOR_VALUE_CENTER)));
        presence.forEach(p -> p.setValue(Math.abs(p.getValue() - SENSOR_VALUE_CENTER)));

        List<SensorValue> data = new ArrayList<>();
        data.addAll(absence);
        data.addAll(presence);
        SensorBeliefModel<Double> result = KnnUtils.generateKFoldModel(k, data, frame);
        result = generateWeakeningModel(result, data);
        result = new ConsonantBeliefModel<>(result);
        show(result, data);
        showErrors(result, data);
    }

    private static void crossValidation(FrameOfDiscernment frame, List<SensorValue> absence,
                                        List<SensorValue> presence) {
        // extracting cross validation data

        List<SensorValue> data = new ArrayList<>(absence);
        data.addAll(presence);
        List<SensorValue> crossValidation = KnnUtils.extractSubList(absence, TRAINING_SET_RATIO);
        crossValidation.addAll(KnnUtils.extractSubList(presence, TRAINING_SET_RATIO));

        //creating training set
        List<SensorValue> trainingSet = new ArrayList<>();
        trainingSet.addAll(absence);
        trainingSet.addAll(presence);

        SensorBeliefModel<Double> result = KnnUtils.getBestKnnBeliefForAlphaAndK(frame, trainingSet,
                crossValidation, (a, b) -> Math.abs(a - b));
        result = generateWeakeningModel(result, data);
        result = new ConsonantBeliefModel<>(result);
        show(result, data);
        showErrors(result, data);
    }


    private static void printAbsenceAndPresenceDependingOnAlpha() throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");

        String absenceFile = "samples/sample-1/sensor-2/absence-motion2.json";
        String presenceFile = "samples/sample-1/sensor-2/presence-motion2.json";
        List<SensorValue> absence = getPoints("absence", absenceFile);
        List<SensorValue> presence = getPoints("presence", presenceFile);

        List<SensorValue> trainingSet = new ArrayList<>();
        trainingSet.addAll(absence);
        trainingSet.addAll(presence);

        double sensorValue = 2170.0;
        int k = 10;
        for (int i = 0; i <= 100; i++) {
            double alpha = i * 0.01;
            KnnBelief<Double> model = new KnnBelief<>(trainingSet, k, alpha, frame,
                    l -> l.stream().reduce(Combinations::dempster).get(), (a,b) -> Math.abs(a - b));
            System.out.println(alpha + ";"
                    + model.toMass(sensorValue).get(frame.toStateSet("absence"))
                    + ";" + model.toMass(sensorValue).get(frame.toStateSet("presence")));
        }
    }

    private static List<SensorValue> toDerivative(List<SensorValue> sortedPoints) {
        List<SensorValue> points = new ArrayList<>(sortedPoints.size() - 1);
        for (int i = 1; i < sortedPoints.size(); i++) {
            SensorValue point = sortedPoints.get(i);
            SensorValue prevPoint = sortedPoints.get(i - 1);
            double absoluteDerivative = Math.abs(point.getValue() - prevPoint.getValue())
                    / (point.getTimestamp() - prevPoint.getTimestamp());
            points.add(new SensorValue(point.getSensor(), point.getLabel(), point.getTimestamp(),
                    absoluteDerivative));
        }
        return points;
    }


    private static void showBestMatchWithWeakening(FrameOfDiscernment frame,
                                                   List<SensorValue> trainingSet,
                                                   List<SensorValue> crossValidation) {
        KnnBelief<Double> bestModel = KnnUtils.getBestKnnBelief(frame, trainingSet, crossValidation,
                ALPHA, (a,b) -> Math.abs(a - b));
        show(generateWeakeningModel(bestModel, crossValidation), trainingSet);
    }

    private static void displayTabsDependingOnK(FrameOfDiscernment frame,
                                                List<SensorValue> trainingSet) {
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        JTabbedPane tabbedPane = new JTabbedPane();

        windowFrame.setContentPane(tabbedPane);
        windowFrame.setSize(800, 500);
        windowFrame.setVisible(true);

        double min = trainingSet.stream().mapToDouble(SensorValue::getValue).min().getAsDouble();
        double max = trainingSet.stream().mapToDouble(SensorValue::getValue).max().getAsDouble();
        double margin = (max - min) * 0.1;

        for (int neighborCount = 1; neighborCount <= trainingSet.size(); neighborCount++) {
            KnnBelief<Double>  beliefModel = new KnnBelief<>(trainingSet, neighborCount, ALPHA, frame,
                    KnnUtils::optimizedDuboisAndPrade, (a,b) -> Math.abs(a - b));

            JPanel panel = JfreeChartDisplay1D.getChartPanel(beliefModel, 2000, min - margin, max + margin);
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
        show(bestModel, trainingSet);
    }

    private static void show(SensorBeliefModel<Double> model, List<SensorValue> points) {
        double min = points.stream().mapToDouble(SensorValue::getValue).min().getAsDouble();
        double max = points.stream().mapToDouble(SensorValue::getValue).max().getAsDouble();
        ChartPanel chartPanel = JfreeChartDisplay1D.getChartPanel(model, 2000, min , max);
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        windowFrame.setContentPane(chartPanel);
        windowFrame.pack();
        windowFrame.setVisible(true);
    }


    private static DiscountingBeliefModel generateWeakeningModel(SensorBeliefModel<Double> model,
                                                                 List<SensorValue> crossValidation) {
        DecisionStrategy decisionStrategy =
                new CriteriaDecisionStrategy(0.5, 0.6, 0.7, Criteria::betP);
        WeakeningFunction weakeningFunction = new WeakeningFunction();
        DiscountingBeliefModel discountingBeliefModel =
                new DiscountingBeliefModel(model, weakeningFunction);

        Map<String, Double> standardDevs = getStandardDevs(crossValidation);
        for (SensorValue sensorValue : crossValidation) {
            Decision decision = decisionStrategy.decide(
                    discountingBeliefModel.toMass(sensorValue.getValue()));
            StateSet actualDecision = decision.getStateSet();
            StateSet expectedDecision = model.getFrame().toStateSet(sensorValue.getLabel());

            if (!actualDecision.includesOrEquals(expectedDecision)) {
                System.out.println("bad decision for value: " + sensorValue.getValue());
                //FIXME This weakening is arbitrary
                double weakening = decision.getConfidence() - 0.5;
                weakeningFunction.addWeakeningPoint(sensorValue.getValue(), weakening,
                        standardDevs.get(sensorValue.getLabel()));
            }
        }

        return discountingBeliefModel;
    }

    private static void showErrors(SensorBeliefModel<Double> model,
                                   List<SensorValue> crossValidation) {
        DecisionStrategy decisionStrategy =
                new CriteriaDecisionStrategy(0.5, 0.6, 0.7, Criteria::betP);
        int errorCount = 0;
        int imprecisionCount = 0;

        for (SensorValue sensorValue : crossValidation) {
            Decision decision = decisionStrategy.decide(
                    model.toMass(sensorValue.getValue()));
            StateSet actualDecision = decision.getStateSet();
            StateSet expectedDecision = model.getFrame().toStateSet(sensorValue.getLabel());

            if (!actualDecision.includesOrEquals(expectedDecision)) {
                //System.out.println("bad decision for value: " + sensorValue.getValue());
                errorCount++;
            }
            else if (!actualDecision.equals(expectedDecision)) {
                imprecisionCount++;
            }
        }
        System.out.println(errorCount + " errors out of " + crossValidation.size()
                + " (" + (double)errorCount * 100 / crossValidation.size() +" %) tested point" +
                " with decision algorithm.");
        System.out.println(imprecisionCount + " imprecision out of " + crossValidation.size()
                + " (" + (double)imprecisionCount * 100 / crossValidation.size() +" %) tested point" +
                " with decision algorithm.");
    }


    /**
     * Generate a map which gives the standard deviation for each label in the training set.
     * @param trainingSet
     * @return
     */
    private static Map<String,Double> getStandardDevs(List<SensorValue> trainingSet) {
        Map<String, Double> stdDevs = new HashMap<>();
        List<String> labels = trainingSet.stream()
                .map(SensorValue::getLabel).distinct()
                .collect(Collectors.toList());

        for (String label : labels) {
            double[] values = trainingSet.stream()
                    .filter(p -> p.getLabel().equals(label))
                    .mapToDouble(SensorValue::getValue)
                    .toArray();
            double average = Arrays.stream(values).average().getAsDouble();
            double squareAverage = Arrays.stream(values).map(a -> a * a)
                    .average().getAsDouble();

            stdDevs.put(label, Math.sqrt(squareAverage - (average * average)));
        }
        return stdDevs;
    }

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

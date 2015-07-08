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
import fr.inria.tacoma.knn.core.KnnFactory;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.util.DiscountingBeliefModel;
import fr.inria.tacoma.knn.util.KnnUtils;
import org.jfree.chart.ChartPanel;

import javax.swing.*;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Main1D {

    public static final double ALPHA = 0.2;
    public static final double TRAINING_SET_RATIO = 0.6;
    public static final int SENSOR_VALUE_CENTER = 2048;
    public static final String ABSENCE_SAMPLE = "samples/sample-1/sensor-1/absence-motion1.json";
    public static final String PRESENCE_SAMPLE = "samples/sample-1/sensor-1/presence-motion1.json";

    public static void main(String[] args) throws IOException {
//        System.in.read();
//        showGenerated();
        testKfold(5);
//        printAbsenceAndPresenceDependingOnAlpha();
//        testNormal();
    }

    private static SensorBeliefModel<Double> modelForFiles(FrameOfDiscernment frame,
                                                           KnnFactory factory, int foldNb,
                                                           String... files)
            throws IOException {
        List<LabelledPoint<Double>> data = parseData(frame, files);
        Kfold<Double> kfold = new Kfold<>(factory, data, foldNb);

        return kfold.generateModel();
    }

    private static List<LabelledPoint<Double>> parseData(FrameOfDiscernment frame,
                                                         String... files) throws IOException {
        List<LabelledPoint<Double>> data = new ArrayList<>();
        for (String file : files) {
            int fileOffSet = file.lastIndexOf('/');
            String label = file.substring(fileOffSet + 1, file.indexOf('-', fileOffSet));
            data.addAll(getPoints(label, file, frame));
        }
        return data;
    }

    private static void showGenerated() throws IOException {
        FrameOfDiscernment frame= FrameOfDiscernment.newFrame("generated", "A", "B", "C");

        KnnFactory<Double> factory = KnnFactory.getDoubleKnnFactory(frame);
//        factory.setCombination(functions -> functions.stream().reduce(Combinations::dempster).get());
        SensorBeliefModel<Double> model = modelForFiles(frame,
                factory, 5,
                "samples/sample-6/sensor-0/A-sensor0.json",
                "samples/sample-6/sensor-0/B-sensor0.json",
                "samples/sample-6/sensor-0/C-sensor0.json");
//        model = new ConsonantBeliefModel<>(model);
        show(model, "generated kfold", 0 , 1000);

    }

    private static void testNormal() throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");

        String absenceFile = ABSENCE_SAMPLE;
        String presenceFile = PRESENCE_SAMPLE;
        List<LabelledPoint<Double>> absence = getPoints("absence", absenceFile, frame);
        List<LabelledPoint<Double>> presence = getPoints("presence", presenceFile, frame);

        System.out.println(
                "using " + absenceFile + " for absence and " + presenceFile + " for presence");

//        absence.forEach(p -> p.setValue(Math.abs(p.getValue() - SENSOR_VALUE_CENTER)));
//        presence.forEach(p -> p.setValue(Math.abs(p.getValue() - SENSOR_VALUE_CENTER)));
        crossValidation(frame, absence, presence);
    }

    private static void testKfold(int k) throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");

        String absenceFile = ABSENCE_SAMPLE;
        String presenceFile = PRESENCE_SAMPLE;

        System.out.println("using " + absenceFile + " for absence and " + presenceFile +
                " for presence");
        KnnFactory<Double> factory = KnnFactory.getDoubleKnnFactory(frame);
        SensorBeliefModel<Double> model = modelForFiles(frame, factory, k, ABSENCE_SAMPLE,
                PRESENCE_SAMPLE);
        show(model, "k fold generated", 0, 4096);
        List<LabelledPoint<Double>> validation = parseData(frame, ABSENCE_SAMPLE,
                PRESENCE_SAMPLE);
        showErrors(model, validation);
        show(generateWeakeningModel(model, validation), "weakened", 0, 4096);
    }

    private static void crossValidation(FrameOfDiscernment frame, List<LabelledPoint<Double>> absence,
                                        List<LabelledPoint<Double>> presence) {
        // extracting cross validation data

        List<LabelledPoint<Double>> crossValidation = KnnUtils.extractSubList(absence, TRAINING_SET_RATIO);
        crossValidation.addAll(KnnUtils.extractSubList(presence, TRAINING_SET_RATIO));

        List<LabelledPoint<Double>> data = new ArrayList<>(absence);
        data.addAll(presence);

        //creating training set
        List<LabelledPoint<Double>> trainingSet = new ArrayList<>();
        trainingSet.addAll(absence);
        trainingSet.addAll(presence);
        KnnFactory<Double> factory = KnnFactory.getDoubleKnnFactory(frame);
        factory.setCombination(functions -> functions.stream().reduce(Combinations::dempster).get());
        SensorBeliefModel<Double> result =
                KnnUtils.getBestKnnBeliefForAlphaAndK(factory, trainingSet, crossValidation);
        show(result, data, "cross validation");
//        result = generateWeakeningModel(result, data);
//        result = new ConsonantBeliefModel<>(result);
//        System.out.println("error for best model : " + KnnUtils.error(data, result));
//        show(result, data, "cross validation");
//        showErrors(result, data);
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



    private static void displayTabsDependingOnK(FrameOfDiscernment frame,
                                                List<LabelledPoint<Double>> trainingSet) {
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        JTabbedPane tabbedPane = new JTabbedPane();

        windowFrame.setContentPane(tabbedPane);
        windowFrame.setSize(800, 500);
        windowFrame.setVisible(true);

        double min = trainingSet.stream().mapToDouble(LabelledPoint::getValue).min().getAsDouble();
        double max = trainingSet.stream().mapToDouble(LabelledPoint::getValue).max().getAsDouble();
        double margin = (max - min) * 0.1;

        KnnFactory<Double> factory = KnnFactory.getDoubleKnnFactory(frame);

        Map<String, Double> gammas = KnnUtils
                .generateGammaProvider((a, b) -> Math.abs(a - b), trainingSet);

        for (int neighborCount = 1; neighborCount <= Math.min(100, trainingSet.size()); neighborCount++) {
            KnnBelief<Double>  beliefModel =
                    factory.newKnnBelief(trainingSet, gammas, neighborCount, ALPHA);

            JPanel panel = JfreeChartDisplay1D.getChartPanel(beliefModel, 2000, min - margin,
                    max + margin, "belief mapping, " + KnnUtils.error(trainingSet, beliefModel));
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
    private static List<LabelledPoint<Double>> getPoints(String label, String file,
                                                         FrameOfDiscernment frame) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, true);
        InputStream resourceAsStream = Thread.currentThread().getContextClassLoader()
                .getResourceAsStream(file);
        MappingIterator<SensorValue> iterator = mapper.readValues(
                new JsonFactory().createParser(resourceAsStream),
                SensorValue.class);
        List<LabelledPoint<Double>> points = new ArrayList<>();
        while (iterator.hasNext()) {
            SensorValue next = iterator.next();
            next.setLabel(label);
            next.setStateSet(frame.toStateSet(next.getLabel()));
            points.add(next);
        }
        return points;
    }


    private static void show(SensorBeliefModel<Double> model, List<LabelledPoint<Double>> points,
                             String title) {
        double min = points.stream().mapToDouble(LabelledPoint::getValue).min().getAsDouble();
        double max = points.stream().mapToDouble(LabelledPoint::getValue).max().getAsDouble();
        show(model, title, min, max);
    }

    private static void show(SensorBeliefModel<Double> model, String title, double min,
                             double max) {
        ChartPanel chartPanel = JfreeChartDisplay1D.getChartPanel(model, 2000, min, max);
        JFrame windowFrame = new JFrame();
        windowFrame.setTitle(title);
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        windowFrame.setContentPane(chartPanel);
        windowFrame.pack();
        windowFrame.setVisible(true);
    }


    private static DiscountingBeliefModel generateWeakeningModel(SensorBeliefModel<Double> model,
                                                                 List<LabelledPoint<Double>> crossValidation) {
        DecisionStrategy decisionStrategy =
                new CriteriaDecisionStrategy(0.5, 0.6, 0.7, Criteria::betP);
        WeakeningFunction weakeningFunction = new WeakeningFunction();
        DiscountingBeliefModel discountingBeliefModel =
                new DiscountingBeliefModel(model, weakeningFunction);

        Map<String, Double> standardDevs = getStandardDevs(crossValidation);

        //FIXME This weakening is arbitrary
        final double baseWeakening = 0.1;
        double weakening = 0.1;
        for (LabelledPoint<Double> sensorValue : crossValidation) {
            Decision decision = decisionStrategy.decide(
                    discountingBeliefModel.toMass(sensorValue.getValue()));
            StateSet actualDecision = decision.getStateSet();
            StateSet expectedDecision = model.getFrame().toStateSet(sensorValue.getLabel());

            if (!actualDecision.includesOrEquals(expectedDecision)) {

                System.out.println("bad decision for value: " + sensorValue.getValue());
                System.out.println(
                        "actualDecision=" + actualDecision + " real = " + expectedDecision);
                System.out.println(decision);
                weakeningFunction.putWeakeningPoint(sensorValue.getValue(), weakening,
                        standardDevs.get(sensorValue.getLabel()));
            }
        }

        return discountingBeliefModel;
    }

    private static void showErrors(SensorBeliefModel<Double> model,
                                   List<LabelledPoint<Double>> crossValidation) {
        DecisionStrategy decisionStrategy =
                new CriteriaDecisionStrategy(0.5, 0.6, 0.7, Criteria::betP);
        int errorCount = 0;
        int imprecisionCount = 0;

        for (LabelledPoint<Double> sensorValue : crossValidation) {
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
    private static Map<String,Double> getStandardDevs(List<LabelledPoint<Double>> trainingSet) {
        Map<String, Double> stdDevs = new HashMap<>();
        List<String> labels = trainingSet.stream()
                .map(LabelledPoint::getLabel).distinct()
                .collect(Collectors.toList());

        for (String label : labels) {
            double[] values = trainingSet.stream()
                    .filter(p -> p.getLabel().equals(label))
                    .mapToDouble(LabelledPoint::getValue)
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

        public void putWeakeningPoint(double center, double weakening, double amplitude) {
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

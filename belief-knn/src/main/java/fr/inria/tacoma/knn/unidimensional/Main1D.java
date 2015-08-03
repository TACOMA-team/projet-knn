package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MutableMass;
import fr.inria.tacoma.bft.criteria.Criteria;
import fr.inria.tacoma.bft.decision.CriteriaDecisionStrategy;
import fr.inria.tacoma.bft.decision.Decision;
import fr.inria.tacoma.bft.decision.DecisionStrategy;
import fr.inria.tacoma.bft.sensorbelief.FusionSensorBeliefModel;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.core.*;
import fr.inria.tacoma.knn.util.ConsonantBeliefModel;
import fr.inria.tacoma.knn.util.DiscountingBeliefModel;
import fr.inria.tacoma.knn.util.KnnUtils;
import org.jfree.chart.ChartPanel;

import javax.swing.*;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
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
//        testKfold(3);
//        printAbsenceAndPresenceDependingOnAlpha();
//        testNormal();
//        activityLevel();
//        gammaOptimization();
//        gammaOptimization2();
//        long before = System.nanoTime();
//        activityLevelAlphaGamma();
//        System.out.println("elapsed: " + (double) (System.nanoTime() - before) / 1e9 + " s");
//        showErrorForActivityLevel();
    }

    private static void showGenerated() throws IOException {
        FrameOfDiscernment frame= FrameOfDiscernment.newFrame("generated", "A", "B", "C");

        KnnFactory<Double> factory = KnnFactory.getDoubleKnnFactory(frame);
//        factory.setCombination(functions -> functions.stream().reduce(Combinations::dempster).get());
        List<LabelledPoint<Double>> data = KnnUtils.parseData(frame,
                "samples/sample-6/sensor-0/A-sensor0.json",
                "samples/sample-6/sensor-0/B-sensor0.json",
                "samples/sample-6/sensor-0/C-sensor0.json");
        SensorBeliefModel<Double> model = Kfold.generateModel(factory, data, 3);
        model = new ConsonantBeliefModel<>(model);
        show(model, "generated kfold", 0, 1000);
//        PrintStream printStream = new PrintStream(new FileOutputStream("generated2_Dempster_consonant.csv"));
//        BeliefModelPrinter.printSensorBeliefAsCSV(model, printStream, 0, 1000, 1000);
        System.out.println("global error : " + KnnUtils.error(data, model));
    }

    private static void testNormal() throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");

        String absenceFile = ABSENCE_SAMPLE;
        String presenceFile = PRESENCE_SAMPLE;
        List<LabelledPoint<Double>> absence = KnnUtils.getPoints("absence", absenceFile, frame);
        List<LabelledPoint<Double>> presence = KnnUtils.getPoints("presence", presenceFile, frame);

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
//        factory.setCombination(functions -> functions.stream().reduce(Combinations::dempster).get());
        List<LabelledPoint<Double>> data = KnnUtils.parseData(frame,
                ABSENCE_SAMPLE, PRESENCE_SAMPLE);
        SensorBeliefModel<Double> model = Kfold.generateModel(factory, data, k);

        PrintStream printStream = new PrintStream(new FileOutputStream("presence_diff_DuboisPrade.csv"));
        BeliefModelPrinter.printSensorBeliefAsCSV(model, printStream, - 2048, 2048, 2000);

        show(model, "k fold generated", 0, 4096);

        System.out.println("global error : " + KnnUtils.error(data, model));
        Errors.showErrors(model, data);
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
        DiscountingBeliefModel.WeakeningFunction weakeningFunction = new DiscountingBeliefModel.WeakeningFunction();
        DiscountingBeliefModel discountingBeliefModel =
                new DiscountingBeliefModel(model, weakeningFunction);

        Map<String, Double> standardDevs = getStandardDevs(crossValidation);

        boolean wrongDecisionOccured = true;
        while (wrongDecisionOccured) {
            wrongDecisionOccured = false;
            for (LabelledPoint<Double> measure : crossValidation) {
                StateSet expectedDecision = model.getFrame().toStateSet(measure.getLabel());
                StateSet actualDecision;

                Decision decision = decisionStrategy.decide(
                        discountingBeliefModel.toMass(measure.getValue()));
                actualDecision = decision.getStateSet();

                if (!actualDecision.includesOrEquals(expectedDecision)) {

                    System.out.println("bad decision for value: " + measure.getValue());
                    System.out.println(
                            "actualDecision=" + actualDecision + " real = " + expectedDecision);
                    System.out.println(decision);

                    double weakening = 0.1 + weakeningFunction.getWeakening(measure.getValue());
                    System.out.println("applying discounting " + weakening);

                    weakeningFunction.putWeakeningPoint(measure.getValue(), weakening,
                            standardDevs.get(measure.getLabel()));
                    wrongDecisionOccured = true;
                }
            }

        }

        return discountingBeliefModel;
    }


    /**
     * Generate a map which gives the standard deviation for each label in the training set.
     * @param trainingSet
     * @return
     */
    public static Map<String,Double> getStandardDevs(List<LabelledPoint<Double>> trainingSet) {
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


    private static void activityLevel() throws IOException {
        FrameOfDiscernment frame =
                FrameOfDiscernment.newFrame("activity", "absence", "weak", "average", "max");
        KnnFactory<Double> factory = KnnFactory.getDoubleKnnFactory(frame);
        int windowSize = 5;
//        List<LabelledPoint<Double>> data = KnnUtils.parseData(frame,
//                Features.slidingAverage(windowSize),
//                "samples/sample-7/son-yoann/absence-yoannd1.json",
//                "samples/sample-7/son-yoann/average-yoannd1.json",
//                "samples/sample-7/son-yoann/max-yoannd1.json",
//                "samples/sample-7/son-yoann/weak-yoannd1.json");
//        List<LabelledPoint<Double>> data = KnnUtils.parseData(frame,
//                "samples/sample-7/soundphidget/absence-soundphidget.json",
//                "samples/sample-7/soundphidget/weak-soundphidget.json",
//                "samples/sample-7/soundphidget/average-soundphidget.json",
//                "samples/sample-7/soundphidget/max-soundphidget.json");
        Function<List<LabelledPoint<Double>>, List<LabelledPoint<Double>>> modifier = Features::normalized;
        modifier = Features.slidingAverage(10);
        List<LabelledPoint<Double>> data = KnnUtils.parseData(frame,
                modifier,
                "samples/sample-7/motion0/absence-motion0.json",
                "samples/sample-7/motion0/weak-motion0.json",
                "samples/sample-7/motion0/average-motion0.json",
                "samples/sample-7/motion0/max-motion0.json");
//        SensorBeliefModel<Double> model = Kfold.generateModel(factory, data, 3);
//        show(model, data, "activity level dubois prade");
//        show(new ConsonantBeliefModel<>(model), data, "consonant activity dubois prade");

        factory = KnnFactory.getDoubleDempsterFactory(frame);
        SensorBeliefModel<Double> dempster = Kfold.generateModel(factory, data, 3);
        show(dempster, data, "activity level dempster");
        show(new ConsonantBeliefModel<>(dempster), data, "consonant activity dempster");

    }



    private static void showErrorForActivityLevel() throws IOException {
        FrameOfDiscernment frame =
                FrameOfDiscernment.newFrame("activity", "absence", "weak", "average", "max");
//
//        List<LabelledPoint<Double>> sound = KnnUtils.parseData(frame,
////                Features::log,
//                "samples/sample-7/soundphidget/absence-soundphidget.json",
//                "samples/sample-7/soundphidget/weak-soundphidget.json",
//                "samples/sample-7/soundphidget/average-soundphidget.json",
//                "samples/sample-7/soundphidget/max-soundphidget.json");
        List<LabelledPoint<Double>> motion0 = KnnUtils.parseData(frame,
                "samples/sample-7/motion0/absence-motion0.json",
                "samples/sample-7/motion0/weak-motion0.json",
                "samples/sample-7/motion0/average-motion0.json",
                "samples/sample-7/motion0/max-motion0.json");
        List<LabelledPoint<Double>> motion1 = KnnUtils.parseData(frame,
                "samples/sample-7/motion1/absence-motion1.json",
                "samples/sample-7/motion1/weak-motion1.json",
                "samples/sample-7/motion1/average-motion1.json",
                "samples/sample-7/motion1/max-motion1.json");
        KnnFactory<Double> factory = KnnFactory.getDoubleDempsterFactory(frame);
//        SensorBeliefModel<Double> soundModel = KfoldAlphaGammas.generateModel(factory, sound, 3);
        SensorBeliefModel<Double> motion0Model = KfoldAlphaGammas.generateModel(factory, motion0, 3);
        SensorBeliefModel<Double> motion1Model = KfoldAlphaGammas.generateModel(factory, motion1, 3);

        Map<SensorBeliefModel<Double>,List<LabelledPoint<Double>>> models = new HashMap<>();

//        models.put(soundModel, sound);
        models.put(motion0Model, motion0);
        models.put(motion1Model, motion1);


        Errors.showGlobalErrors(models);

        Errors.showTimeLine(models);

    }

    private static class FakeFusionModel extends FusionSensorBeliefModel<Double> {

        public FakeFusionModel(
                SensorBeliefModel<Double> underlyingModel, double maxTime) {
            super(underlyingModel, maxTime);
        }

        @Override
        public MutableMass toMass(Double sensorValue) {
            return this.toMass(sensorValue, 1.0);
        }
    }
}

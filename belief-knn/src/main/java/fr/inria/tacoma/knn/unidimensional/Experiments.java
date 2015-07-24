package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.core.KnnBelief;
import fr.inria.tacoma.knn.core.KnnFactory;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.util.ConsonantBeliefModel;
import fr.inria.tacoma.knn.util.KnnUtils;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;

public class Experiments {

    public static void main(String[] args) throws IOException {
        generateFiles();
    }

    public static void generateFiles() throws IOException {
//        printGeneratedDataSets();
//
//        printPresenceAbsenceSample();

//        printAlphas();
    }

    private static void printPresenceAbsenceSample() throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");

        //presence_DuboisPrade.csv
        KnnFactory<Double> factory = KnnFactory.getDoubleKnnFactory(frame);
        List<LabelledPoint<Double>> data = KnnUtils.parseData(frame,
                "samples/sample-1/sensor-1/absence-motion1.json",
                "samples/sample-1/sensor-1/presence-motion1.json");
        printModel("presence_DuboisPrade.csv", factory, data, 0, 4096, 1000);

        //presence_Dempster.csv
        factory.setCombination(masses -> masses.stream().reduce(Combinations::dempster).get());
        printModel("presence_Dempster.csv", factory, data, 0, 4096, 1000);


        //presence_diff_Dubois_Prade.csv
        data = KnnUtils.parseData(frame,
                list -> {
                    for (int i = list.size() - 1; i > 1; i--) {
                        LabelledPoint<Double> previous = list.get(i - 1);
                        LabelledPoint<Double> current = list.get(i);
                        list.get(i).setValue(current.getValue() - previous.getValue());
                    }

                    list.remove(0);
                    return list;
                },
                "samples/sample-1/sensor-1/absence-motion1.json",
                "samples/sample-1/sensor-1/presence-motion1.json");

        factory = KnnFactory.getDoubleKnnFactory(frame);
        printModel("presence_diff_Dubois_Prade.csv", factory, data, -2048, 2048, 1000);
    }

    private static void printGeneratedDataSets() throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("generated", "A", "B");

        List<LabelledPoint<Double>> data = KnnUtils.parseData(frame,
                "samples/sample-6/sensor-0/A-sensor0.json",
                "samples/sample-6/sensor-0/B-sensor0.json");
        KnnFactory<Double> factory = KnnFactory.getDoubleKnnFactory(frame);
        printModel("generated_DuboisPrade.csv", factory, data, 0, 1000, 1000);


        //generated_Dempster.csv
        factory.setCombination(masses -> masses.stream().reduce(Combinations::dempster).get());
        printModel("generated_Dempster.csv", factory, data, 0, 1000, 1000);


        //generated2_DuboisPrade.csv
        frame = FrameOfDiscernment.newFrame("generated", "A", "B", "C");
        data = KnnUtils.parseData(frame,
                "samples/sample-6/sensor-0/A-sensor0.json",
                "samples/sample-6/sensor-0/B-sensor0.json",
                "samples/sample-6/sensor-0/C-sensor0.json");

        factory = KnnFactory.getDoubleKnnFactory(frame);
        printModel("generated2_DuboisPrade.csv", factory, data, 0, 1000, 1000);

        //generated2_DuboisPrade_consonant.csv
        printModel("generated2_DuboisPrade_consonant.csv", factory, data, 0, 1000, 1000, true);

        //generated2_Dempster.csv
        factory.setCombination(masses -> masses.stream().reduce(Combinations::dempster).get());
        printModel("generated2_Dempster.csv", factory, data, 0, 1000, 1000);


        //generated2_Dempster_consonant.csv
        printModel("generated2_Dempster_consonant.csv", factory, data, 0, 1000, 1000, true);
    }

    private static void printModel(String name, KnnFactory<Double> factory,
                                   List<LabelledPoint<Double>> data, int min, int max,
                                   int numPoints) throws IOException {
        printModel(name,  factory, data, min, max, numPoints, false);
    }


    private static void printModel(String name, KnnFactory<Double> factory,
                                   List<LabelledPoint<Double>> data, int min, int max,
                                   int numPoints, boolean consonant) throws IOException {
        // generated_DuboisPrade.csv
        System.out.println(name);
        System.out.println("-------------------------");

        SensorBeliefModel<Double> model = Kfold.generateModel(factory, data, 3);
        if(consonant) {
            model = new ConsonantBeliefModel<>(model);
        }
        PrintStream printStream = new PrintStream(new FileOutputStream(name));
        BeliefModelPrinter.printSensorBeliefAsCSV(model, printStream, min, max, numPoints);
        System.out.println("global error : " + KnnUtils.error(data, model));
    }

    private static void printAlphas() throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");

        //presence_DuboisPrade.csv
        KnnFactory<Double> factory = KnnFactory.getDoubleKnnFactory(frame);
        List<LabelledPoint<Double>> data = KnnUtils.parseData(frame,
                "samples/sample-1/sensor-1/absence-motion1.json",
                "samples/sample-1/sensor-1/presence-motion1.json");

        PrintStream printStream = new PrintStream(new FileOutputStream("alpha.csv"));
        printStream.println("alpha;error");

//        List<LabelledPoint<Double>> crossValidation =

        for (double alpha = 0; alpha < 1; alpha += 0.01){
            KnnBelief<Double> beliefModel = factory.newKnnBelief(data,
                    KnnUtils.generateGammaProvider((a, b) -> Math.abs(a - b), data), 2, alpha);
            System.out.println(KnnUtils.error(data, beliefModel));
        }
    }

}

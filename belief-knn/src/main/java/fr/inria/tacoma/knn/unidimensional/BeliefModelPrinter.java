package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;

import java.io.PrintStream;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.IntStream;

public class BeliefModelPrinter {

    public static void printSensorBeliefAsCSV(SensorBeliefModel<Double> beliefModel,
                                                PrintStream printStream, double min, double max,
                                              int numPoints) {
        TreeMap<Double, MassFunction> massFunctionSet = new TreeMap<>();
        Map<Double, MassFunction> syncMap = Collections.synchronizedMap(massFunctionSet);
        IntStream.range(0, numPoints).parallel()
                .mapToDouble(x -> min + (x * ((max - min) / numPoints)))
                .forEach(value -> syncMap.put(value, beliefModel.toMass(value)));

        printSensorBeliefAsCSV(massFunctionSet, beliefModel.getFrame(), printStream);
    }

    private static void printSensorBeliefAsCSV(TreeMap<Double, MassFunction> massFunctionSet,
                                                     FrameOfDiscernment frame,
                                                     PrintStream printStream) {

        printStream.print("sensor value;");
        for (int i = 1; i <= frame.card(); i++) {
            for (StateSet stateSet : frame.getStateSetsWithCard(i)) {
                printStream.print(stateSet + ";");
            }
        }
        printStream.println();
        for (Map.Entry<Double, MassFunction> entry : massFunctionSet.entrySet()) {
            printStream.print(entry.getKey() + ";");
            MassFunction massFunction = entry.getValue();
            for (int i = 1; i <= frame.card(); i++) {
                frame.getStateSetsWithCard(i).stream().sorted()
                        .forEach(stateSet -> printStream.print(massFunction.get(stateSet) + ";"));
            }
            printStream.println();
        }
    }
}

package fr.inria.tacoma.knn.launcher;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.core.Kfold;
import fr.inria.tacoma.knn.core.KnnFactory;
import fr.inria.tacoma.knn.core.LabelledPoint;
import fr.inria.tacoma.knn.unidimensional.BeliefModelPrinter;
import fr.inria.tacoma.knn.unidimensional.SensorValue;
import fr.inria.tacoma.knn.util.KnnUtils;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.type.FileArgumentType;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.*;
import java.util.function.BiFunction;


public class Launcher {

    private static BiFunction<Double, Double, Double> doubleDistance =(a, b) -> Math.abs(a - b);

    enum Mode {
        findBest {
            @Override
            public SensorBeliefModel<Double> run(List<LabelledPoint<Double>> points,
                                                 FrameOfDiscernment frame, Namespace arguments) {
                Combination combination = arguments.get("combination");
                KnnFactory<Double> factory = combination.getFactory(frame);
                int folds = arguments.get("fold");

                return Kfold.generateModel(factory, points, folds);
            }
        },
        fixedParams {
            @Override
            public SensorBeliefModel<Double> run(List<LabelledPoint<Double>> points,
                                                 FrameOfDiscernment frame, Namespace arguments) {
                Combination combination = arguments.get("combination");
                KnnFactory<Double> factory = combination.getFactory(frame);
                int k = arguments.get("k");
                double alpha = arguments.get("alpha");
                Map<String, Double> gammaProvider = KnnUtils.generateGammaProvider(doubleDistance,
                        points);

                return factory.newKnnBelief(points, gammaProvider, k, alpha);
            }
        };

        abstract public SensorBeliefModel<Double> run(List<LabelledPoint<Double>> points,
                                                      FrameOfDiscernment frame, Namespace arguments);
    }

    enum Combination {
        dubois {
            @Override
            public KnnFactory<Double> getFactory(FrameOfDiscernment frame) {
                return KnnFactory.getDoubleKnnFactory(frame);
            }
        }, dempster {
            @Override
            public KnnFactory<Double> getFactory(FrameOfDiscernment frame) {
                return KnnFactory.getDoubleDempsterFactory(frame);
            }
        };

        public abstract KnnFactory<Double> getFactory(FrameOfDiscernment frame);
    }

    public static void main(String[] args) throws ArgumentParserException, IOException {
        ArgumentParser parser = getArgumentParser();
        Namespace parsedArgs = parser.parseArgsOrFail(args);

        PrintStream output = System.out;
        if(parsedArgs.get("output") != null) {
            output = new PrintStream(new FileOutputStream((File)parsedArgs.get("output")));
        }


        File inputFile = parsedArgs.get("inputFile");
        FrameOfDiscernment frame = getFrame("frame", inputFile);
        List<LabelledPoint<Double>> points = parsePoints(frame, inputFile);

        Mode mode = parsedArgs.get("mode");

        SensorBeliefModel<Double> result = mode.run(points, frame, parsedArgs);


        Double min =  parsedArgs.get("min");
        if(min  == null) {
            min = getMin(points);
        }

        Double max = parsedArgs.get("min");
        if(max == null) {
           max = getMax(points);
        }

        BeliefModelPrinter.printSensorBeliefAsCSV(result, output, min, max,
                parsedArgs.get("numPoints"));
    }

    private static double getMax(List<LabelledPoint<Double>> points) {
        return points.stream().mapToDouble(LabelledPoint::getValue).max().getAsDouble();
    }

    private static double getMin(List<LabelledPoint<Double>> points) {
        return points.stream().mapToDouble(LabelledPoint::getValue).min().getAsDouble();
    }

    private static FrameOfDiscernment getFrame(String name, File inputFile) throws IOException {
        Set<String> classes = new HashSet<>();
        Iterable<CSVRecord> records = CSVFormat.EXCEL.withHeader().parse(new FileReader(inputFile));

        for (CSVRecord record : records) {
            String clazz = record.get("class");
            classes.add(clazz);
        }
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame(name, classes);

        return frame;
    }

    private static List<LabelledPoint<Double>> parsePoints(FrameOfDiscernment frame, File inputFile) throws IOException {

        Reader in = new FileReader(inputFile);
        List<LabelledPoint<Double>> sensorValues = new ArrayList<>();
        Iterable<CSVRecord> records = CSVFormat.EXCEL.withHeader().parse(in);

        for (CSVRecord record : records) {
            String name = record.get("name");
            double value = Double.valueOf(record.get("value"));
            String clazz = record.get("class");
            Double timestamp = Double.valueOf(record.get("timestamp"));
            SensorValue sensorValue = new SensorValue(name, clazz, timestamp, value,
                    frame.toStateSet(clazz));
            sensorValues.add(sensorValue);
        }

        return sensorValues;
    }

    private static ArgumentParser getArgumentParser() {
        ArgumentParser parser = ArgumentParsers.newArgumentParser("knn")
                .description("find the best knn model");

        parser.addArgument("inputFile")
                .type(new FileArgumentType())
                .required(true);

        parser.addArgument("--mode")
                .type(Mode.class)
                .choices(Mode.values())
                .setDefault(Mode.findBest)
                .help("mode to use, default is " + Mode.findBest  + ".\n" +
                        Mode.findBest +": find the best combination of alpha and k \n" +
                        Mode.fixedParams+ ": do not try to find parameters, " +
                        "just use the provided alpha and k");
        parser.addArgument("-c", "--combination")
                .type(Combination.class)
                .choices(Combination.values())
                .setDefault(Combination.dempster)
                .help("type of combination to use");
        parser.addArgument("-o", "--output")
                .type(new FileArgumentType())
                .help("output for the csv, default is stdout");
        parser.addArgument("-k")
                .type(Integer.class)
                .help("neighbor number to use (for " + Mode.fixedParams + " only)");
        parser.addArgument("-a", "--alpha")
                .type((_parser, arg, value) -> {
                    Double doubleValue = Double.valueOf(value);
                    if (doubleValue < 0 || doubleValue > 0.9) {
                        throw new ArgumentParserException(
                                "alpha must be between 0 and 0.9, current value is " + value,
                                _parser);
                    }
                    return doubleValue;
                })
                .help("alpha to use (for " + Mode.fixedParams + " only)");
        parser.addArgument("-f", "--fold")
                .type(Integer.class)
                .setDefault(3)
                .help("number of folds to use in k-fold.");
        parser.addArgument("-n", "--numPoints").setDefault(1000)
                .type(Integer.class)
                .help("number of point to have in the output csv");
        parser.addArgument("-m", "--min").type(Integer.class)
                .help("minimum sensor value in the output csv. " +
                        "Default is minimum value in the input csv.");
        parser.addArgument("-M", "--max").type(Integer.class)
                .help("maximum sensor value in the output csv. " +
                        "Default is maximum value in the input csv.");
        return parser;
    }
}

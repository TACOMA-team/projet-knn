package fr.inria.tacoma.bft.parser.xml;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.sensorbelief.FusionSensorBeliefModel;
import fr.inria.tacoma.bft.sensorbelief.LinearSensorBeliefModel;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.bft.sensorbelief.SpecificitySensorBeliefModel;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.stream.StreamSource;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Parser for the xml representation of a sensor belief model.
 */
public class XmlModelParser {

    /**
     * This function is a simple helper as it just open an input stream on the
     * file specified by the path and give it to the parse function taking an
     * input stream as argument.
     * @param path path to the file
     * @return a mapping between a sensor id and the model used for this sensor.
     * @throws FileNotFoundException if the file does not exist
     */
    public static Map<String, SensorBeliefModel> parse(String path) throws FileNotFoundException {
        return parse(new FileInputStream(path));
    }

    /**
     * Parse an xml file from an input stream. This xml file must contain a
     * belief model (see the example xml in README)
     * @param stream stream of the xml
     * @return a mapping between a sensor id and the model used for this sensor.
     */
    public static Map<String, SensorBeliefModel> parse(InputStream stream) {
        try {
            DocumentBuilder builder = getDocumentBuilder();
            Document document = builder.parse(stream);
            FrameOfDiscernment frame = parseFrame(document);

            return parseDocument(document, frame);

        } catch (SAXException | ParserConfigurationException | IOException e) {
            e.printStackTrace();//FIXME
        }
        return null;
    }

    private static Map<String, SensorBeliefModel> parseDocument(Document document,
                                                                FrameOfDiscernment frame) {
        Map<String, SensorBeliefModel> beliefMap = new HashMap<>();

        //parse each sensor to get the mapped belief model
        NodeList sensorElements = document.getElementsByTagName("sensor");
        for (int i = 0; i < sensorElements.getLength(); i++) {
            Element sensorElement = (Element)sensorElements.item(i);
            String sensorName = sensorElement.getAttribute("name");

            //retrieves the belief model used by the sensor
            Element beliefModel = document.getElementById(sensorElement.getAttribute("belief"));

            SensorBeliefModel model = parseSensorBelief(beliefModel, frame);
            beliefMap.put(sensorName, model);
        }

        return beliefMap;
    }

    /**
     * @return the document builder used by xml belief models
     * @throws SAXException
     * @throws ParserConfigurationException
     */
    private static DocumentBuilder getDocumentBuilder()
            throws SAXException, ParserConfigurationException {
        SchemaFactory schemaFactory =
                SchemaFactory.newInstance(XMLConstants.W3C_XML_SCHEMA_NS_URI);
        Schema schema = schemaFactory.newSchema(new StreamSource(XmlModelParser.class
                .getClassLoader()
                .getResourceAsStream("belief-model.xsd")));
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setSchema(schema);
        factory.setNamespaceAware(true);
        return factory.newDocumentBuilder();
    }

    /**
     * Parses a single sensor belief.
     * @param beliefElement belief xml tag
     * @param frame frame of discernment for the current model
     * @return a new sensor belief
     */
    private static SensorBeliefModel parseSensorBelief(Element beliefElement,
                                                       FrameOfDiscernment frame) {
        NodeList points = beliefElement.getElementsByTagName("point");
        LinearSensorBeliefModel baseModel = new LinearSensorBeliefModel(frame);

        for (int i = 0; i < points.getLength(); i++) {
            Element point = (Element)points.item(i);
            Element valueElement = (Element)point.getElementsByTagName("value").item(0);

            double value = Double.parseDouble(valueElement.getTextContent());
            baseModel.addInterpolationFunction(value, parseInterpolationMass(frame, point));
        }

        return withOption(beliefElement, baseModel);
    }

    /**
     * Parses the interpolation mass function for a single point.
     * @param frame The frame of discernment used by the model.
     * @param point the point for which we want the interpolation mass function.
     * @return a mass function used as interpolation point.
     */
    private static MassFunction parseInterpolationMass(FrameOfDiscernment frame, Element point) {
        NodeList massList = point.getElementsByTagName("mass");

        MassFunction interpolationMassFunction = new MassFunctionImpl(frame);
        for (int j = 0; j < massList.getLength(); j++) {
            Element massElement = (Element)massList.item(j);
            interpolationMassFunction.set(
                    frame.toStateSet(massElement.getAttribute("set").split(" ")),
                    Double.parseDouble(massElement.getTextContent()));
        }
        return interpolationMassFunction;
    }

    /**
     * Takes a linear sensor belief model and adds the option given in the xml element.
     * For instance, if the option is tempo-fusion, it will return a FusionSensorBeliefModel.
     * @param beliefElement belief root element
     * @param baseModel linear sensor belief model which is used
     * @return the new model
     */
    private static SensorBeliefModel<Double> withOption(Element beliefElement,
                                                LinearSensorBeliefModel baseModel) {
        NodeList options = beliefElement.getElementsByTagName("option");

        if(options.getLength() > 0) { //if we have an option in the xml, use it
            Element optionElement = (Element) options.item(0);
            switch (optionElement.getAttribute("name")) {
                case "tempo-specificity":
                    return new SpecificitySensorBeliefModel<>(baseModel,
                            Double.parseDouble(optionElement.getTextContent()));
                case "tempo-fusion":
                    return new FusionSensorBeliefModel<>(baseModel,
                            Double.parseDouble(optionElement.getTextContent()));
                default:
                    throw new UnsupportedOperationException("option " +
                            optionElement.getAttribute("name") + " is unknown.");
            }
        }
        else {
            return baseModel;
        }
    }

    /**
     * Parses the frame of discernment used in the xml document.
     * @param document xml document to parse
     * @return the frame of discernment for the model being parsed
     */
    private static FrameOfDiscernment parseFrame(Document document) {
        Element frameNode = (Element)document.getElementsByTagName("frame").item(0);
        NodeList stateNodes = frameNode.getElementsByTagName("state");
        List<String> states = new ArrayList<>();
        for (int i = 0; i < stateNodes.getLength(); i++) {
            states.add(stateNodes.item(i).getTextContent());
        }
        return FrameOfDiscernment.newFrame(frameNode.getAttribute("name"), states);
    }

}

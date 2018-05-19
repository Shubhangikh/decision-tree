package com.utd.ml.dtree;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class ID3 {
    private static Map<String, Integer> attributeIndexMap = new HashMap<>();
    private static int nodeLabel = 1;

    public static void main(String[] args) {
        List<List<String>> trainingInstances = new ArrayList<>();
        List<List<String>> testInstances = new ArrayList<>();
        List<List<String>> validationInstances = new ArrayList<>();
        List<String> attributes = new ArrayList<>();
        String targetAttribute = null;
        try {
            int l = Integer.parseInt(args[0]);
            int k = Integer.parseInt(args[1]);
            List<String> trainingData = Files.readAllLines(Paths.get(args[2]));
            List<String> validationData = Files.readAllLines(Paths.get(args[3]));
            List<String> testData = Files.readAllLines(Paths.get(args[4]));
            String toPrint = args[5];

            for (int i = 0; i < trainingData.size(); i++) {
                String[] values = trainingData.get(i).split(",");
                if (i == 0) {
                    int j = 0;
                    for (; j < values.length - 1; j++) {
                        attributes.add(values[j]);
                        attributeIndexMap.put(values[j], j);
                    }
                    attributeIndexMap.put(values[j], j);
                    targetAttribute = values[j];
                } else {
                    trainingInstances.add(Arrays.asList(values));
                }
            }

            for (int i = 1; i < testData.size(); i++) {
                String[] values = testData.get(i).split(",");
                testInstances.add(Arrays.asList(values));
            }
            for (int i = 1; i < validationData.size(); i++) {
                String[] values = validationData.get(i).split(",");
                validationInstances.add(Arrays.asList(values));
            }
            DecisionTree<String> tree1 = new DecisionTree<>();
            nodeLabel = 1;
            tree1.setRoot(buildTree(trainingInstances, targetAttribute, attributes, tree1.getRoot(), true));
            DecisionTree<String> tree2 = new DecisionTree<>();
            nodeLabel = 1;
            tree2.setRoot(buildTree(trainingInstances, targetAttribute, attributes, tree2.getRoot(), false));

            if (toPrint.equalsIgnoreCase("yes")) {
                System.out.println("\n\nDecision tree build by applying Information Gain Heuristic:");
                printTree(tree1.getRoot(), 0, toPrint);
            }
            System.out.print("\nClassification Accuracy of decision tree build by applying Information Gain Heuristic: " + calculateAccuracy(tree1.getRoot(), testInstances));
            DecisionTree.Node<String> node1 = pruning(l, k, tree1.getRoot(), validationInstances);
            System.out.print("\nAccuracy of decision tree build by applying Information Gain Heuristic post pruning: " + calculateAccuracy(node1, testInstances));

            if (toPrint.equalsIgnoreCase("yes")) {
                System.out.println("\n\nDecision tree build by applying Impurity Variance Heuristic:");
                printTree(tree2.getRoot(), 0, toPrint);
            }
            System.out.print("\nClassification Accuracy of decision tree build by applying Impurity Variance Heuristic: " + calculateAccuracy(tree2.getRoot(), testInstances));
            DecisionTree.Node<String> node2 = pruning(l, k, tree2.getRoot(), validationInstances);
            System.out.print("\nAccuracy of decision tree build by applying Impurity Variance Heuristic post pruning: " + calculateAccuracy(node2, testInstances));

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double calculateAccuracy(DecisionTree.Node<String> root, List<List<String>> testInstances) {
        int correctPredictionCount = 0;
        DecisionTree.Node<String> tempNode = null;
        for (List<String> instance : testInstances) {
            tempNode = root;

            while (true) {
                if (attributeIndexMap.containsKey(tempNode.getData())) {
                    int index = attributeIndexMap.get(tempNode.getData());
                    if (instance.get(index).equals(tempNode.getLeftBranchValue())) {
                        if (tempNode.getLeft() == null) {
                            break;
                        }
                        tempNode = tempNode.getLeft();
                    } else {
                        if (tempNode.getRight() == null) {
                            break;
                        }
                        tempNode = tempNode.getRight();
                    }
                } else
                    break;
            }
            if (instance.get(instance.size() - 1).equals(tempNode.getData())) {
                correctPredictionCount++;
            }
        }
        return (double) correctPredictionCount / testInstances.size() * 100;
    }

    private static void printTree(DecisionTree.Node<String> node, int depth, String toPrint) {
        if ("yes".equalsIgnoreCase(toPrint)) {
            if (node.left == null && node.right == null)
                return;
            if (node.left != null) {
                int i = 0;
                while (i < depth) {
                    i++;
                    System.out.print("| ");
                }
                System.out.print(node.data + " =  0 : ");
                if (node.left.left == null && node.left.right == null)
                    System.out.print(" " + node.left.data);
                System.out.println();
                printTree(node.left, depth + 1, toPrint);
            }
            if (node.right != null) {
                int i = 0;
                while (i < depth) {
                    i++;
                    System.out.print("| ");
                }
                System.out.print(node.data + " =  1 : ");
                if (node.right.left == null && node.right.right == null)
                    System.out.print(" " + node.right.data);
                System.out.println();
                printTree(node.right, depth + 1, toPrint);
            }
        }


    }

    private static DecisionTree.Node<String> buildTree(List<List<String>> instances, String targetAttribute, List<String> attributes,
                                                       DecisionTree.Node<String> node, boolean igHeuristic) {

        if (attributes == null || attributes.size() == 0) {
            return null;
        }
        boolean allPositive = true;
        boolean allNegative = true;
        for (List<String> instance : instances) {
            if (instance.get(instance.size() - 1).equalsIgnoreCase("0")) {
                allPositive = false;
                break;
            }
        }
        for (List<String> instance : instances) {
            if (instance.get(instance.size() - 1).equalsIgnoreCase("1")) {
                allNegative = false;
                break;
            }
        }
        if (allPositive) {
            node.setData("1");
            return node;
        }
        if (allNegative) {
            node.setData("0");
            return node;
        }
        String attribute = determineAttrWithMaxGain(instances, attributes, igHeuristic);

        // Set properties of node object
        node.setData(attribute);
        node.setLeftBranchValue("0");
        node.setRightBranchValue("1");
        node.setLabel(nodeLabel++);
        List<List<List<String>>> tempList = divideInstancesByTargetAttribute(instances, instances.get(0).size() - 1);
        int countOfClassZero = tempList.get(1).size();
        int countOfClassOne = tempList.get(0).size();
        node.setSplitProportion(new int[]{countOfClassZero, countOfClassOne});

        List<String> leftAttributes = new ArrayList<>(attributes);
        List<String> rightAttributes = new ArrayList<>(attributes);
        leftAttributes.remove(attribute);
        rightAttributes.remove(attribute);

        boolean sameTargetForAll0 = true;
        boolean sameTargetForAll1 = true;
        String targetValueFor0 = null;
        String targetValueFor1 = null;

        List<List<String>> zeroValuedInstances = new ArrayList<>();
        List<List<String>> oneValuedInstances = new ArrayList<>();
        int index = attributeIndexMap.get(attribute);
        for (List<String> instance : instances) {
            if ("0".equals(instance.get(index))) {
                zeroValuedInstances.add(instance);
            } else {
                oneValuedInstances.add(instance);
            }
        }

        if (zeroValuedInstances.size() > 0) {
            int sizeOfInnerList = zeroValuedInstances.get(0).size();
            targetValueFor0 = zeroValuedInstances.get(0).get(sizeOfInnerList - 1);
            for (int i = 1; i < zeroValuedInstances.size(); i++) {
                if (!targetValueFor0.equals(zeroValuedInstances.get(i).get(sizeOfInnerList - 1))) {
                    sameTargetForAll0 = false;
                    break;
                }
            }
        }

        if (oneValuedInstances.size() > 0) {
            int sizeOfInnerList = oneValuedInstances.get(0).size();
            targetValueFor1 = oneValuedInstances.get(0).get(sizeOfInnerList - 1);
            for (int j = 1; j < oneValuedInstances.size(); j++) {
                if (!targetValueFor1.equals(oneValuedInstances.get(j).get(sizeOfInnerList - 1))) {
                    sameTargetForAll1 = false;
                    break;
                }
            }
        }

        if (sameTargetForAll0 && sameTargetForAll1) {
            if (zeroValuedInstances.size() > 0) {
                node.setLeft(new DecisionTree.Node<>(targetValueFor0));
            }
            if (oneValuedInstances.size() > 0) {
                node.setRight(new DecisionTree.Node<>(targetValueFor1));
            }
        } else {
            if (sameTargetForAll0) {
                if (zeroValuedInstances.size() > 0) {
                    node.setLeft(new DecisionTree.Node<>(targetValueFor0));
                }
                node.setRight(buildTree(oneValuedInstances, targetAttribute, rightAttributes, new DecisionTree.Node<>(), igHeuristic));
            } else if (sameTargetForAll1) {
                if (oneValuedInstances.size() > 0) {
                    node.setRight(new DecisionTree.Node<>(targetValueFor1));
                }
                node.setLeft(buildTree(zeroValuedInstances, targetAttribute, leftAttributes, new DecisionTree.Node<>(), igHeuristic));
            } else {
                node.setLeft(buildTree(zeroValuedInstances, targetAttribute, leftAttributes, new DecisionTree.Node<>(), igHeuristic));
                node.setRight(buildTree(oneValuedInstances, targetAttribute, rightAttributes, new DecisionTree.Node<>(), igHeuristic));
            }
        }

        return node;
    }

    private static String determineAttrWithMaxGain(List<List<String>> instances, List<String> attributes, boolean igHeuristic) {
        double informationGain = 0.0;
        String bestClassifier = null;
        for (String attribute : attributes) {
            double iGain = calculateGain(attribute, instances, igHeuristic);
            if (informationGain <= iGain) {
                informationGain = iGain;
                bestClassifier = attribute;
            }
        }
        return bestClassifier;
    }

    private static double calculateGain(String attribute, List<List<String>> instances, boolean igHeuristic) {
        int indexOfTargetOutput = instances.get(0).size() - 1;
        int indexOfAttributeValue = attributeIndexMap.get(attribute);
        List<List<List<String>>> allInstances = divideInstancesByTargetAttribute(instances, indexOfTargetOutput);
        int positiveTargetCount = allInstances.get(0).size();
        int negativeTargetCount = allInstances.get(1).size();
        int total = positiveTargetCount + negativeTargetCount;

        int countPos0 = countInstancesByAttributeValue(allInstances.get(0), indexOfAttributeValue, "0");
        int countPos1 = countInstancesByAttributeValue(allInstances.get(0), indexOfAttributeValue, "1");
        int countNeg0 = countInstancesByAttributeValue(allInstances.get(1), indexOfAttributeValue, "0");
        int countNeg1 = countInstancesByAttributeValue(allInstances.get(1), indexOfAttributeValue, "1");

        double factor = 0.0, factorS0 = 0.0, factorS1 = 0.0;
        if (igHeuristic) {
            factor = calculateEntropy(positiveTargetCount, negativeTargetCount);
        } else {
            factor = calculateImpurityVariance(positiveTargetCount, negativeTargetCount);
        }
        if (igHeuristic) {
            factorS0 = calculateEntropy(countPos0, countNeg0);
            factorS1 = calculateEntropy(countPos1, countNeg1);
        } else {
            factorS0 = calculateImpurityVariance(countPos0, countNeg0);
            factorS1 = calculateImpurityVariance(countPos1, countNeg1);
        }

        int count0 = countPos0 + countNeg0;
        int count1 = countPos1 + countNeg1;
        return (factor - (((double) count0 / total) * factorS0 + ((double) count1 / total) * factorS1));
    }

    private static double calculateEntropy(int positiveTargetCount, int negativeTargetCount) {
        if (positiveTargetCount == 0 || negativeTargetCount == 0) {
            return 0.0;
        }
        int totalInstances = positiveTargetCount + negativeTargetCount;
        double posInsProp = (double) positiveTargetCount / totalInstances;
        double negInsProp = (double) negativeTargetCount / totalInstances;
        return (-posInsProp * (Math.log(posInsProp) / Math.log(2)))
                + (-negInsProp * (Math.log(negInsProp) / Math.log(2)));
    }

    private static double calculateImpurityVariance(int positiveTargetCount, int negativeTargetCount) {
        if (positiveTargetCount == 0 || negativeTargetCount == 0) {
            return 0.0;
        }
        int totalInstances = positiveTargetCount + negativeTargetCount;
        double posInsProp = (double) positiveTargetCount / totalInstances;
        double negInsProp = (double) negativeTargetCount / totalInstances;
        return (posInsProp * negInsProp);
    }

    private static List<List<List<String>>> divideInstancesByTargetAttribute(List<List<String>> instances, int index) {
        List<List<String>> positiveInstances = new ArrayList<>();
        List<List<String>> negativeInstances = new ArrayList<>();
        List<List<List<String>>> allInstances = new ArrayList<>();
        for (List<String> instance : instances) {
            if (instance.get(index).equals("0")) {
                negativeInstances.add(instance);
            } else {
                positiveInstances.add(instance);
            }
        }
        allInstances.add(positiveInstances);
        allInstances.add(negativeInstances);
        return allInstances;
    }

    private static int countInstancesByAttributeValue(List<List<String>> instances, int index, String value) {
        int count = 0;
        for (List<String> instance : instances) {
            if (instance.get(index).equals(value)) {
                count++;
            }
        }
        return count;
    }

    private static DecisionTree.Node<String> pruning(int l, int k, DecisionTree.Node<String> node, List<List<String>> validationInstances) {

        DecisionTree.Node<String> dTreeBest = node;
        double accuracy = calculateAccuracy(node, validationInstances);
        for (int i = 0; i < l; i++) {
            Random random = new Random();
            int m = random.nextInt((k - 1) + 1) + 1;
            for (int j = 0; j < m; j++) {
                DecisionTree.Node<String> tempNode = copy(node);
                int numberOfNonLeafNodes = countNonLeafNodes(tempNode);
                int p = random.nextInt((numberOfNonLeafNodes - 1) + 1) + 1;
                double accuracyOfPrunedTree = pruneAndCalculateAccuracy(tempNode, validationInstances, p);

                if (accuracy < accuracyOfPrunedTree) {
                    dTreeBest = tempNode;
                }
            }
        }

        return dTreeBest;
    }

    private static double pruneAndCalculateAccuracy(DecisionTree.Node<String> tempNode, List<List<String>> validationInstances, int p) {
        Stack<DecisionTree.Node<String>> s = new Stack<>();
        DecisionTree.Node<String> element = tempNode;
        s.add(element);
        while (!s.isEmpty()) {
            element = s.pop();
            if (element.label == p) {
                if (element.left == null && element.right == null) {
                    break;
                }
                element.left = null;
                element.right = null;
                element.setData(String.valueOf(element.getSplitProportion()[0] > element.getSplitProportion()[1] ? 0 : 1));
                break;
            }
            if (element.right != null) {
                s.add(element.right);
            }
            if (element.left != null) {
                s.add(element.left);
            }
        }
        return calculateAccuracy(tempNode, validationInstances);
    }

    private static int countNonLeafNodes(DecisionTree.Node<String> tempNode) {
        if (tempNode == null || (tempNode.left == null && tempNode.right == null)) {
            return 0;
        }
        return 1 + countNonLeafNodes(tempNode.left) + countNonLeafNodes(tempNode.right);
    }

    private static DecisionTree.Node<String> copy(DecisionTree.Node<String> n1) {
        if (n1 == null) {
            return null;
        }
        DecisionTree.Node<String> node = new DecisionTree.Node<>(n1.data, n1.leftBranchValue, n1.rightBranchValue, null, null, n1.label, n1.splitProportion);
        node.left = copy(n1.left);
        node.right = copy(n1.right);
        return node;
    }

}

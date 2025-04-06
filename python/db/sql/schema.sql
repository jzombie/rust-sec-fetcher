-- MySQL dump 10.13  Distrib 8.0.34, for macos13 (arm64)
--
-- Host: 127.0.0.1    Database: us_gaap
-- ------------------------------------------------------
-- Server version	8.1.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ofss_category`
--

DROP TABLE IF EXISTS `ofss_category`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ofss_category` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `group_id` int unsigned NOT NULL,
  `category_name` varchar(72) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `category_name_group_UNIQUE` (`category_name`,`group_id`),
  KEY `fk_group_id_idx` (`group_id`),
  CONSTRAINT `fk_group_id` FOREIGN KEY (`group_id`) REFERENCES `ofss_group` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=140 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ofss_category`
--

LOCK TABLES `ofss_category` WRITE;
/*!40000 ALTER TABLE `ofss_category` DISABLE KEYS */;
INSERT INTO `ofss_category` VALUES (63,5,'Accounting Change'),(23,3,'Accounts Payable'),(111,9,'Accounts Payable'),(108,9,'Accounts Receivable'),(4,2,'Accounts Receivable - Current'),(5,2,'Accounts Receivable - Noncurrent'),(25,3,'Accrued Expenses'),(16,2,'Accumulated Depreciation'),(97,7,'Accumulated Other Comprehensive Income'),(114,10,'Acquisition of Business'),(38,4,'Additional Paid-In Capital'),(104,9,'Amortization'),(75,5,'Basic Earnings Per Share Excluding Extraordinary Items'),(76,5,'Basic Earnings Per Share Including Extraordinary Items'),(74,5,'Basic Weighted Average Shares'),(12,2,'Buildings - Gross'),(31,3,'Capital Lease Obligations'),(1,2,'Cash'),(2,2,'Cash & Equivalents'),(119,11,'Cash Dividends Paid - Common'),(131,11,'Cash Dividends Paid - Preferred'),(128,8,'Cash Interest Paid'),(129,8,'Cash Taxes Paid'),(139,5,'Common Dividends'),(37,4,'Common Stock'),(122,11,'Common Stock, Net'),(100,7,'Convertible Instrument Equity Adjustments'),(46,5,'Cost of Revenue'),(27,3,'Current Portion of Long Term Debt and Capital Leases'),(28,3,'Customer Advances'),(32,3,'Deferred Income Tax'),(106,9,'Deferred Revenue'),(105,9,'Deferred Taxes'),(102,9,'Depreciation'),(49,5,'Depreciation/Amortization'),(80,5,'Diluted Earnings Per Share Excluding Extraordinary Items'),(81,5,'Diluted Earnings Per Share Including Extraordinary Items'),(78,5,'Diluted Net Income'),(79,5,'Diluted Weighted Average Shares'),(77,5,'Dilution Adjustment'),(64,5,'Discontinued Operations'),(91,7,'Dividends Declared - Common Stock'),(92,7,'Dividends Declared - Preferred Stock'),(41,4,'Employee Stock Ownership Plan Debt Guarantee'),(61,5,'Equity In Affiliates'),(85,6,'Equity Method OCI Component'),(65,5,'Extraordinary Item'),(82,6,'Foreign Currency Translation Adjustment'),(127,8,'Foreign Exchange Effects'),(57,5,'Gain (Loss) on Sale of Assets'),(68,5,'General Partners\' Distributions'),(17,2,'Goodwill'),(72,5,'Income Available to Common Shareholders Excluding Extraordinary'),(73,5,'Income Available to Common Shareholders Including Extraordinary'),(18,2,'Intangibles'),(71,5,'Interest Adjustment - Primary Earnings Per Share'),(55,5,'Interest Expense - Non-Operating'),(50,5,'Interest Expense - Operating'),(52,5,'Interest Expense (Income) - Operating'),(56,5,'Interest Income - Non-Operating'),(51,5,'Interest/Investment Income - Operating'),(109,9,'Inventories'),(7,2,'Inventories - Finished Goods'),(8,2,'Inventories - Work in Progress'),(93,7,'Issuance of Common Stock'),(13,2,'Land/Improvements - Gross'),(30,3,'Long Term Debt'),(125,11,'Long Term Debt Reduction'),(126,11,'Long Term Debt, Net'),(19,2,'Long Term Investments - Other'),(14,2,'Machinery/Equipment - Gross'),(33,3,'Minority Interest'),(60,5,'Minority Interest'),(69,5,'Miscellaneous Earnings Adjustment'),(89,7,'Net Income'),(103,9,'Net Income/Starting Line'),(45,5,'Net Sales'),(98,7,'Noncontrolling Interest Changes'),(26,3,'Notes Payable/Short Term Debt'),(20,2,'Notes Receivable - Long Term'),(110,9,'Other Assets'),(44,4,'Other Comprehensive Income'),(88,6,'Other Comprehensive Income (After Tax)'),(87,6,'Other Comprehensive Income (Before Tax)'),(11,2,'Other Current Assets'),(29,3,'Other Current Liabilities'),(101,7,'Other Equity Adjustments'),(118,11,'Other Financing Cash Flow'),(117,10,'Other Investing Cash Flow'),(112,9,'Other Liabilities'),(22,2,'Other Long Term Assets'),(34,3,'Other Long Term Liabilities'),(107,9,'Other Non-Cash Items'),(58,5,'Other Non-Operating Income (Expense)'),(54,5,'Other Operating Expenses'),(130,5,'Other Operating Income'),(15,2,'Other Property/Plant/Equipment - Gross'),(24,3,'Payable/Accrued'),(84,6,'Pension and Postretirement Benefit Adjustments'),(67,5,'Preferred Dividends'),(90,7,'Preferred Dividends'),(36,4,'Preferred Stock - Non Redeemable'),(9,2,'Prepaid Expenses'),(70,5,'Pro Forma Adjustment'),(132,2,'Property/Plant/Equipment - Net'),(59,5,'Provision for Income Taxes'),(113,10,'Purchase of Fixed Assets'),(133,10,'Purchase of Intangibles'),(116,10,'Purchase of Investments'),(134,2,'Receivables, Net, Current'),(6,2,'Receivables, Other'),(99,7,'Redeemable Noncontrolling Interest Adjustments'),(35,4,'Redeemable Preferred Stock'),(94,7,'Repurchase of Common Stock'),(121,11,'Repurchase/Retirement of Common'),(135,11,'Repurchase/Retirement of Convertible Preferred'),(136,11,'Repurchase/Retirement of Non-Convertible Preferred'),(48,5,'Research & Development'),(10,2,'Restricted Cash - Current'),(21,2,'Restricted Cash - Long Term'),(39,4,'Retained Earnings (Accumulated Deficit)'),(120,11,'Sale/Issuance of Common'),(137,11,'Sale/Issuance of Preferred'),(115,10,'Sale/Maturity of Investment'),(47,5,'Selling/General/Administrative Expenses'),(123,11,'Short Term Debt Issued'),(124,11,'Short Term Debt, Net'),(3,2,'Short Term Investments'),(95,7,'Stock-Based Compensation'),(66,5,'Tax on Extraordinary Items'),(138,2,'Total Assets'),(43,4,'Translation Adjustment'),(40,4,'Treasury Stock - Common'),(96,7,'Treasury Stock Activity'),(62,5,'U.S. GAAP Adjustment'),(86,6,'Unrealized Derivative Gains (Losses)'),(42,4,'Unrealized Gain (Loss)'),(83,6,'Unrealized Gain (Loss) on Available-for-Sale Securities'),(53,5,'Unusual Expense (Income)');
/*!40000 ALTER TABLE `ofss_category` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ofss_group`
--

DROP TABLE IF EXISTS `ofss_group`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ofss_group` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `parent_group_id` int unsigned DEFAULT NULL,
  `group_name` varchar(45) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `group_name_UNIQUE` (`group_name`),
  UNIQUE KEY `group_parent_UNIQUE` (`parent_group_id`,`id`),
  KEY `fk_parent_group_id_idx` (`parent_group_id`),
  CONSTRAINT `fk_parent_group_id` FOREIGN KEY (`parent_group_id`) REFERENCES `ofss_group` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ofss_group`
--

LOCK TABLES `ofss_group` WRITE;
/*!40000 ALTER TABLE `ofss_group` DISABLE KEYS */;
INSERT INTO `ofss_group` VALUES (1,NULL,'Balance Sheet'),(2,1,'Assets'),(3,1,'Liabilities'),(4,1,'Shareholders\' Equity'),(5,NULL,'Income Statement'),(6,NULL,'Comprehensive Income'),(7,NULL,'Equity Statement'),(8,NULL,'Cash Flow'),(9,8,'Operating Activities'),(10,8,'Investing'),(11,8,'Financing');
/*!40000 ALTER TABLE `ofss_group` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `us_gaap_balance_type`
--

DROP TABLE IF EXISTS `us_gaap_balance_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `us_gaap_balance_type` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `balance` varchar(7) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `balance_UNIQUE` (`balance`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `us_gaap_balance_type`
--

LOCK TABLES `us_gaap_balance_type` WRITE;
/*!40000 ALTER TABLE `us_gaap_balance_type` DISABLE KEYS */;
/*!40000 ALTER TABLE `us_gaap_balance_type` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `us_gaap_period_type`
--

DROP TABLE IF EXISTS `us_gaap_period_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `us_gaap_period_type` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `period_type` varchar(8) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `period_type_UNIQUE` (`period_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `us_gaap_period_type`
--

LOCK TABLES `us_gaap_period_type` WRITE;
/*!40000 ALTER TABLE `us_gaap_period_type` DISABLE KEYS */;
/*!40000 ALTER TABLE `us_gaap_period_type` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `us_gaap_statement_type`
--

DROP TABLE IF EXISTS `us_gaap_statement_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `us_gaap_statement_type` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `statement_type` varchar(20) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `statement_type_UNIQUE` (`statement_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `us_gaap_statement_type`
--

LOCK TABLES `us_gaap_statement_type` WRITE;
/*!40000 ALTER TABLE `us_gaap_statement_type` DISABLE KEYS */;
/*!40000 ALTER TABLE `us_gaap_statement_type` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `us_gaap_tag`
--

DROP TABLE IF EXISTS `us_gaap_tag`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `us_gaap_tag` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `balance_type_id` int unsigned DEFAULT NULL,
  `period_type_id` int unsigned DEFAULT NULL,
  `label` text,
  `description` text,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `name_UNIQUE` (`name`),
  KEY `fk_balance_type_id_idx` (`balance_type_id`),
  KEY `fk_period_type_id_idx` (`period_type_id`),
  CONSTRAINT `fk_balance_type_id` FOREIGN KEY (`balance_type_id`) REFERENCES `us_gaap_balance_type` (`id`),
  CONSTRAINT `fk_period_type_id` FOREIGN KEY (`period_type_id`) REFERENCES `us_gaap_period_type` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `us_gaap_tag`
--

LOCK TABLES `us_gaap_tag` WRITE;
/*!40000 ALTER TABLE `us_gaap_tag` DISABLE KEYS */;
/*!40000 ALTER TABLE `us_gaap_tag` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `us_gaap_tag_ofss_category`
--

DROP TABLE IF EXISTS `us_gaap_tag_ofss_category`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `us_gaap_tag_ofss_category` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `us_gaap_tag_id` int unsigned NOT NULL,
  `ofss_category_id` int unsigned NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  KEY `fk_ofss_category_id_idx` (`ofss_category_id`),
  KEY `fk_us_gaap_tag_id_idx` (`us_gaap_tag_id`),
  CONSTRAINT `fk_ofss_category_id` FOREIGN KEY (`ofss_category_id`) REFERENCES `ofss_category` (`id`),
  CONSTRAINT `fk_us_gaap_tag_id` FOREIGN KEY (`us_gaap_tag_id`) REFERENCES `us_gaap_tag` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `us_gaap_tag_ofss_category`
--

LOCK TABLES `us_gaap_tag_ofss_category` WRITE;
/*!40000 ALTER TABLE `us_gaap_tag_ofss_category` DISABLE KEYS */;
/*!40000 ALTER TABLE `us_gaap_tag_ofss_category` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `us_gaap_tag_statement_type`
--

DROP TABLE IF EXISTS `us_gaap_tag_statement_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `us_gaap_tag_statement_type` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `us_gaap_tag_id` int unsigned NOT NULL,
  `us_gaap_statement_type_id` int unsigned NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `tag_statement_UNIQUE` (`us_gaap_tag_id`,`us_gaap_statement_type_id`),
  KEY `fk_statement_id_idx` (`us_gaap_statement_type_id`),
  CONSTRAINT `fk_statement_id` FOREIGN KEY (`us_gaap_statement_type_id`) REFERENCES `us_gaap_statement_type` (`id`),
  CONSTRAINT `fk_tag_id` FOREIGN KEY (`us_gaap_tag_id`) REFERENCES `us_gaap_tag` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `us_gaap_tag_statement_type`
--

LOCK TABLES `us_gaap_tag_statement_type` WRITE;
/*!40000 ALTER TABLE `us_gaap_tag_statement_type` DISABLE KEYS */;
/*!40000 ALTER TABLE `us_gaap_tag_statement_type` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-05 22:24:48

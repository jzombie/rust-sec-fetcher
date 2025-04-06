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
  `ofss_id` int NOT NULL,
  `category_name` varchar(72) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`),
  UNIQUE KEY `ofss_id_UNIQUE` (`ofss_id`),
  UNIQUE KEY `category_name_group_UNIQUE` (`category_name`,`group_id`),
  KEY `fk_group_id_idx` (`group_id`),
  CONSTRAINT `fk_group_id` FOREIGN KEY (`group_id`) REFERENCES `ofss_group` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

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
-- Table structure for table `us_gaap_tag`
--

DROP TABLE IF EXISTS `us_gaap_tag`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `us_gaap_tag` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(128) NOT NULL,
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
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-05 22:01:27

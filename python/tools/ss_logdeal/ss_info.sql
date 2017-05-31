/*
Navicat MySQL Data Transfer

Target Server Type    : MYSQL
Target Server Version : 50173
File Encoding         : 65001

Date: 2017-05-31 09:45:24
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for ss_info
-- ----------------------------
DROP TABLE IF EXISTS `ss_info`;
CREATE TABLE `ss_info` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `time` int(11) DEFAULT NULL,
  `dst` varchar(255) CHARACTER SET utf8 DEFAULT NULL,
  `dst_port` int(255) DEFAULT NULL,
  `req` varchar(255) CHARACTER SET utf8 DEFAULT NULL,
  `req_port` int(11) DEFAULT NULL,
  `add_time` int(11) DEFAULT NULL,
  `type` varchar(255) CHARACTER SET utf8 DEFAULT '',
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=86 DEFAULT CHARSET=latin1;
SET FOREIGN_KEY_CHECKS=1;
